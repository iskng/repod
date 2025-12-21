use anyhow::{Context, Result};
use chrono::Local;
use clap::Parser;
use copypasta::{ClipboardContext, ClipboardProvider};
use crossterm::style::Stylize;
use crossterm::{
    event::{read, Event, KeyCode},
    terminal,
};
use dirs;
use git2::Repository;
use globset::{Glob, GlobSet, GlobSetBuilder};
use ignore::{DirEntry, WalkBuilder};
use indicatif::{MultiProgress, ParallelProgressIterator, ProgressBar, ProgressStyle};
use infer;
use memmap2::Mmap;
use parking_lot::Mutex;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::ffi::OsStr;
use std::process::Command;
use std::{
    fs::{self, File},
    io::{BufReader, Read, Write},
    path::Path,
    path::PathBuf,
    sync::Arc,
    time::Instant,
};
use tempfile::TempDir;
use tiktoken_rs::o200k_base;

mod tree;
use tree::DirectoryTree;

const LARGE_FILE_THRESHOLD: u64 = 1024 * 1024; // 1MB
const CHUNK_SIZE: usize = 100;
const BINARY_CHECK_SIZE: usize = 8192; // Increased binary check size
const TEXT_THRESHOLD: f32 = 0.3; // Maximum ratio of non-text bytes allowed
const EMPTY_TREE_HASH: &str = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"; // Git's canonical empty tree

// Common text file extensions that we definitely want to include
const TEXT_EXTENSIONS: &[&str] = &[
    // Programming languages
    "rs",
    "py",
    "js",
    "ts",
    "java",
    "c",
    "cpp",
    "h",
    "hpp",
    "cs",
    "go",
    "rb",
    "php",
    "scala",
    "kt",
    "kts",
    "swift",
    "m",
    "mm",
    "r",
    "pl",
    "pm",
    "t",
    "sh",
    "bash",
    "zsh",
    "fish",
    // Web
    "html",
    "htm",
    "css",
    "scss",
    "sass",
    "less",
    "jsx",
    "tsx",
    "vue",
    "svelte",
    // Data/Config
    "json",
    "yaml",
    "yml",
    "toml",
    "xml",
    "csv",
    "ini",
    "conf",
    "config",
    "properties",
    // Documentation
    "md",
    "markdown",
    "rst",
    "txt",
    "asciidoc",
    "adoc",
    "tex",
    // Other
    "sql",
    "graphql",
    "proto",
    "cmake",
    "make",
    "dockerfile",
    "editorconfig",
    "gitignore",
];

// File patterns that should always be excluded
const EXCLUDED_PATTERNS: &[&str] = &[
    ".git/",
    "node_modules/",
    "target/",
    "build/",
    "dist/",
    "bin/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".tox/",
    ".venv/",
    "venv/",
    "env/",
    ".env/",
    ".next/",
    ".nuxt/",
    ".cache/",
    ".parcel-cache/",
    ".turbo/",
    ".vercel/",
    ".output/",
    "coverage/",
    ".nyc_output/",
    ".eggs/",
    "*.egg-info/",
    ".svn/",
    ".hg/",
    ".DS_Store",
    ".idea/",
    ".vs/",
    ".vscode/",
    ".gradle/",
    "out/",
    "tmp/",
    ".tiktoken",
    ".bin",
    ".pack",
    ".idx",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Cargo.lock",
    "poetry.lock",
    "Pipfile.lock",
    "composer.lock",
    "Gemfile.lock",
    "go.sum",
    "mix.lock",
    "flake.lock",
    "pubspec.lock",
    "packages.lock.json",
];

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Git repository URL, path to CSV file, or nothing to use current directory
    #[arg(index = 1)]
    input: Option<String>,

    /// Output directory path
    #[arg(short, long, default_value = "output")]
    output_dir: String,

    /// Repository types to filter files (e.g., rs, py, js, ts)
    /// Can specify multiple times for multiple types
    #[arg(short = 't', long, value_parser = parse_repo_type, value_delimiter = ',')]
    repo_types: Vec<RepoType>,

    /// GitHub personal access token for private repositories
    #[arg(short = 'p', long)]
    github_token: Option<String>,

    /// SSH key path (defaults to ~/.ssh/id_rsa)
    #[arg(long)]
    ssh_key: Option<String>,

    /// SSH key passphrase (if not provided, will prompt if needed)
    #[arg(long)]
    ssh_passphrase: Option<String>,

    /// Open in cursor after cloning
    #[arg(long)]
    open_cursor: bool,

    /// Specific path to clone the repository to
    #[arg(long)]
    at: Option<String>,

    /// Copy output to clipboard instead of saving to file (explicit)
    /// Default behavior is computed: copies for single-target runs unless --write or -o is set
    #[arg(long)]
    copy: bool,

    /// Write output to file instead of copying to clipboard (overrides default copy behavior)
    #[arg(long)]
    write: bool,

    /// Additional folder or path patterns to exclude from processing
    /// Can be specified multiple times or as a comma‑separated list
    #[arg(short = 'e', long = "exclude", value_delimiter = ',')]
    exclude: Vec<String>,

    /// Only include files matching these patterns (supports ** globs)
    /// Can be specified multiple times or as a comma-separated list.
    /// Bare patterns like "*.rs" implicitly match anywhere (we expand to "**/*.rs").
    #[arg(long = "only", value_delimiter = ',')]
    only: Vec<String>,

    /// Only include files under these directories (relative to repo root)
    /// Examples: --only-dir src,docs or --only-dir src/lib,examples
    /// Implemented as globs like "<dir>/**".
    #[arg(long = "only-dir", value_delimiter = ',')]
    only_dirs: Vec<String>,

    /// Stage and commit changes with an AI-generated message (single commit)
    /// Uses Gemini (models/gemini-3-flash-preview) via GEMINI_API_KEY
    #[arg(long)]
    commit: bool,

    /// Analyze changes and propose multiple commits (per-commit confirmations)
    /// Uses Gemini (models/gemini-3-flash-preview) via GEMINI_API_KEY
    #[arg(long = "multi-commit")]
    multi_commit: bool,

    /// Target branch: name or 'auto' to propose a name from changes
    #[arg(long)]
    branch: Option<String>,

    /// After committing, push the current branch to origin (sets upstream if needed)
    #[arg(long)]
    push: bool,

    /// Ask a question about the current repository (--ask "question about repo")
    #[arg(long)]
    ask: Option<String>,

    /// Model to use for --ask: "pro" (gemini-3-pro-preview) or "flash" (gemini-3-flash-preview)
    #[arg(long, default_value = "pro")]
    model: String,
}

#[derive(Debug, Clone)]
enum RepoType {
    Rust,
    Python,
    JavaScript, // Now includes both JS and TS
    Go,
    Java,
}

fn parse_repo_type(s: &str) -> Result<RepoType, String> {
    match s.to_lowercase().as_str() {
        "rs" | "rust" => Ok(RepoType::Rust),
        "py" | "python" => Ok(RepoType::Python),
        "js" | "javascript" | "ts" | "typescript" => Ok(RepoType::JavaScript),
        "go" | "golang" => Ok(RepoType::Go),
        "java" => Ok(RepoType::Java),
        _ => Err(format!("Unknown repository type: {}", s)),
    }
}

fn normalize_rel_path<'a>(path: &'a Path, root: &Path) -> String {
    let rel = path.strip_prefix(root).unwrap_or(path);
    let s = rel.to_string_lossy().replace('\\', "/");
    if s.is_empty() {
        ".".to_string()
    } else {
        s
    }
}

fn build_only_globset(only_patterns: &[String], only_dirs: &[String]) -> Option<GlobSet> {
    let mut builder = GlobSetBuilder::new();
    let mut added = 0usize;

    // Directories: turn into <dir>/** globs
    for d in only_dirs {
        let d = d.trim_matches('/');
        if d.is_empty() {
            continue;
        }
        let pat = format!("{}/**", d);
        if let Ok(glob) = Glob::new(&pat) {
            builder.add(glob);
            added += 1;
        }
    }

    for pat in only_patterns {
        let p = pat.trim();
        if p.is_empty() {
            continue;
        }
        // If pattern has no slash, expand to match anywhere
        let expanded = if p.contains('/') {
            p.to_string()
        } else {
            format!("**/{}", p)
        };
        if let Ok(glob) = Glob::new(&expanded) {
            builder.add(glob);
            added += 1;
        }
    }

    if added == 0 {
        None
    } else {
        builder.build().ok()
    }
}

fn build_exclude_globset(builtin_patterns: &[&str], user_patterns: &[String]) -> Option<GlobSet> {
    let mut builder = GlobSetBuilder::new();
    let mut added = 0usize;

    for pattern in builtin_patterns
        .iter()
        .copied()
        .chain(user_patterns.iter().map(|s| s.as_str()))
    {
        if let Some(glob_pattern) = normalize_exclude_pattern(pattern) {
            if let Ok(glob) = Glob::new(&glob_pattern) {
                builder.add(glob);
                added += 1;
            }
        }
    }

    if added == 0 {
        None
    } else {
        builder.build().ok()
    }
}

fn normalize_exclude_pattern(pattern: &str) -> Option<String> {
    let raw = pattern.trim();
    if raw.is_empty() {
        return None;
    }

    let cleaned = raw.trim_start_matches("./").replace('\\', "/");
    if cleaned.is_empty() {
        return None;
    }

    if cleaned.ends_with('/') {
        let dir = cleaned.trim_end_matches('/');
        if dir.is_empty() {
            return None;
        }
        let dir = dir.trim_start_matches('/');
        if dir.is_empty() {
            return None;
        }
        Some(format!("**/{}/**", dir))
    } else {
        let target = cleaned.trim_start_matches('/');
        if target.starts_with("**/") {
            Some(target.to_string())
        } else {
            Some(format!("**/{}", target))
        }
    }
}

fn get_repo_type_extensions(repo_type: &RepoType) -> &'static [&'static str] {
    match repo_type {
        RepoType::Rust => &["rs", "toml"],
        RepoType::Python => &[
            "py",
            "pyi",
            "pyx",
            "pxd",
            "requirements.txt",
            "setup.py",
            "pyproject.toml",
        ],
        RepoType::JavaScript => &[
            "js",
            "jsx",
            "ts",
            "tsx",
            "json",
            "package.json",
            "tsconfig.json",
            "jsconfig.json",
        ],
        RepoType::Go => &["go", "mod", "sum"],
        RepoType::Java => &["java", "gradle", "maven", "pom.xml", "build.gradle"],
    }
}

#[derive(Default)]
struct ProcessingStats {
    total_files: usize,
    total_tokens: usize,
    clone_time: f64,
    processing_time: f64,
    repo_count: usize,
    binary_files_skipped: usize,
}

struct FileContent {
    path: String,
    content: String,
    token_count: usize,
    metadata_token_count: usize,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Get URLs or use current directory
    let urls = if let Some(input) = &args.input {
        if input.ends_with(".csv") {
            // Check if file exists
            if !Path::new(input).exists() {
                anyhow::bail!("CSV file not found: {}", input);
            }
            read_urls_from_csv(input)?
        } else if input.starts_with("https://") || input.starts_with("git@") {
            vec![input.clone()]
        } else {
            anyhow::bail!(
                "Input must be either a CSV file or a git URL (https:// or git@). Got: {}",
                input
            );
        }
    } else {
        // Use current directory
        vec![".".to_string()]
    };

    // Check for GitHub token in environment if not provided as argument
    let args = if args.github_token.is_none() {
        let mut args = args;
        args.github_token = std::env::var("GITHUB_TOKEN").ok();
        args
    } else {
        args
    };

    let stats = Arc::new(Mutex::new(ProcessingStats::default()));
    let multi_progress = Arc::new(MultiProgress::new());

    // Handle --ask (question about repo) before other flows
    if let Some(question) = &args.ask {
        ensure_gemini_api_key_interactive()?;
        let multi_progress = Arc::new(MultiProgress::new());

        // Resolve target directory:
        // - No input or "." => current dir
        // - HTTPS/SSH URL => clone to temp dir
        // - CSV => not supported
        // - Local path => use it if exists
        let mut _tmp: Option<TempDir> = None;
        let repo_dir: PathBuf = match args.input.as_deref() {
            None | Some(".") => std::env::current_dir()?,
            Some(inp) if inp.ends_with(".csv") => {
                print_warn("--ask does not support CSV inputs; use a single repo or the current directory.");
                return Ok(());
            }
            Some(inp) if inp.starts_with("https://") || inp.starts_with("git@") => {
                let tmp = TempDir::new()?;
                let path = tmp.path().to_path_buf();
                // Clone with progress bars
                let _repo = clone_repository(inp, &path, &args, &multi_progress)
                    .with_context(|| format!("Failed to access repository: {}", inp))?;
                _tmp = Some(tmp);
                path
            }
            Some(local) => {
                let p = PathBuf::from(local);
                if !p.exists() {
                    print_warn(&format!("Path not found: {}", local));
                    return Ok(());
                }
                p
            }
        };

        ask_about_repository(&repo_dir, question, &args, &multi_progress)?;
        return Ok(());
    }

    // Determine if commit is allowed (only for current directory runs)
    let wants_commit = args.commit || args.multi_commit;
    let commit_allowed = wants_commit && urls.len() == 1 && urls[0] == ".";

    // Determine effective copy/write mode
    // Rules:
    // - --write forces writing to file
    // - --copy forces copying to clipboard
    // - Default (neither provided):
    //     * If multiple targets (CSV / multiple URLs): write to file to avoid clipboard races
    //     * Else if output_dir changed from default: write to file
    //     * Else: copy to clipboard
    let multiple_targets = urls.len() > 1;
    let copy_mode_global = if args.write {
        false
    } else if args.copy {
        true
    } else if multiple_targets || args.output_dir != "output" {
        false
    } else {
        true
    };

    // Only create output directory if we're writing to files and not in commit-only mode
    if !copy_mode_global && !commit_allowed {
        fs::create_dir_all(&args.output_dir)?;
    }

    if wants_commit && !commit_allowed {
        println!("--commit/--multi-commit only work on the current directory. Skipping commit.");
    }

    // Process repositories in parallel if there are multiple
    let do_parallel = urls.len() > 1;
    if do_parallel {
        urls.par_iter().try_for_each(|url| {
            process_repository(
                url,
                &args.output_dir,
                Arc::clone(&stats),
                &args,
                copy_mode_global,
                commit_allowed && url == ".",
                Arc::clone(&multi_progress),
            )
        })?;
    } else {
        process_repository(
            &urls[0],
            &args.output_dir,
            Arc::clone(&stats),
            &args,
            copy_mode_global,
            commit_allowed,
            Arc::clone(&multi_progress),
        )?;
    }

    let final_stats = stats.lock();
    if !commit_allowed {
        print_stats(&final_stats);
    }
    Ok(())
}

fn read_urls_from_csv(path: &str) -> Result<Vec<String>> {
    let mut urls = Vec::new();
    let mut reader = csv::Reader::from_path(path)?;
    for result in reader.records() {
        let record = result?;
        if let Some(url) = record.get(0) {
            urls.push(url.to_string());
        }
    }
    Ok(urls)
}

fn read_file_content(path: &Path) -> Result<String> {
    let file = File::open(path)?;
    let metadata = file.metadata()?;

    if metadata.len() > LARGE_FILE_THRESHOLD {
        // Log large file processing
        println!(
            "Processing large file ({:.2} MB): {}",
            (metadata.len() as f64) / 1024.0 / 1024.0,
            path.display()
        );
        // Use memory mapping for large files
        let mmap = unsafe { Mmap::map(&file)? };
        Ok(String::from_utf8_lossy(&mmap).into_owned())
    } else {
        // Use regular reading for small files
        // Read raw bytes first to handle potential non-UTF8 sequences
        let mut buffer = Vec::with_capacity(metadata.len() as usize);
        BufReader::new(file).read_to_end(&mut buffer)?;
        // Convert to string lossily, replacing invalid sequences
        Ok(String::from_utf8_lossy(&buffer).into_owned())
    }
}

fn build_metadata_block(path: &str) -> String {
    let display_name = Path::new(path)
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string());
    format!(
        "<file_info>\npath: {}\nname: {}\n</file_info>\n",
        path, display_name
    )
}

fn process_files_batch(files: &[FileContent], output: &mut dyn Write) -> Result<()> {
    for file in files {
        let metadata_block = build_metadata_block(&file.path);
        output.write_all(metadata_block.as_bytes())?;
        output.write_all(file.content.as_bytes())?;
        output.write_all(b"\n\n")?;
    }
    Ok(())
}

fn handle_auth_error(url: &str, error: &git2::Error) -> anyhow::Error {
    let is_auth_error = error.code() == git2::ErrorCode::Auth
        || error.message().contains("authentication")
        || error.message().contains("authorization");

    if is_auth_error {
        let mut msg = String::from("\nAuthentication failed. To fix this:\n");

        if url.starts_with("https://") {
            msg.push_str(
                "For HTTPS repositories:\n\
                1. Set your GitHub token using one of these methods:\n\
                   - Run with --github-token YOUR_TOKEN\n\
                   - Set the GITHUB_TOKEN environment variable\n\
                2. Ensure your token has the 'repo' scope enabled\n",
            );
        } else if url.starts_with("git@") {
            msg.push_str(
                "For SSH repositories:\n\
                1. Ensure your SSH key is set up correctly:\n\
                   - Default location: ~/.ssh/id_rsa\n\
                   - Or specify with --ssh-key /path/to/key\n\
                2. Verify your SSH key is added to GitHub\n\
                3. Test SSH access: ssh -T git@github.com\n",
            );
        } else {
            msg.push_str(
                "Ensure you're using either:\n\
                - HTTPS URL (https://github.com/org/repo)\n\
                - SSH URL (git@github.com:org/repo)\n",
            );
        }

        anyhow::anyhow!(msg)
    } else {
        anyhow::anyhow!("Git error: {}", error)
    }
}

fn prompt_passphrase(pb: &ProgressBar) -> Result<String> {
    // Pause the spinner while waiting for input
    pb.set_message("Waiting for SSH key passphrase...");
    pb.disable_steady_tick();

    let passphrase = rpassword::prompt_password("Enter SSH key passphrase: ")?;

    // Resume the spinner
    pb.enable_steady_tick(std::time::Duration::from_millis(100));

    Ok(passphrase)
}

fn clone_repository(
    url: &str,
    path: &Path,
    args: &Args,
    multi_progress: &MultiProgress,
) -> Result<Repository> {
    let mut callbacks = git2::RemoteCallbacks::new();
    let mut fetch_options = git2::FetchOptions::new();
    let mut builder = git2::build::RepoBuilder::new();

    // Create progress bar for cloning
    let clone_pb = multi_progress.add(ProgressBar::new_spinner());
    clone_pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg} [{elapsed_precise}]")
            .unwrap()
            .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"),
    );
    clone_pb.enable_steady_tick(std::time::Duration::from_millis(100));

    let result = if url.starts_with("https://") {
        clone_pb.set_message(format!("Connecting to: {}", url));
        // Try without token first for public repos
        let result = builder.clone(url, path);
        if let Err(e) = result {
            if e.code() == git2::ErrorCode::Auth {
                clone_pb.set_message("Repository requires authentication, trying with token...");
                // If auth failed, try with token
                if let Some(token) = &args.github_token {
                    callbacks.credentials(|_url, _username_from_url, _allowed_types| {
                        git2::Cred::userpass_plaintext(token, "x-oauth-basic")
                    });
                    fetch_options.remote_callbacks(callbacks);
                    builder.fetch_options(fetch_options);
                    builder
                        .clone(url, path)
                        .map_err(|e| handle_auth_error(url, &e))
                } else {
                    Err(
                        anyhow::anyhow!(
                            "Repository requires authentication.\n\
                        Please provide a GitHub token using --github-token or set the GITHUB_TOKEN environment variable."
                        )
                    )
                }
            } else {
                Err(handle_auth_error(url, &e))
            }
        } else {
            Ok(result.unwrap())
        }
    } else if url.starts_with("git@") {
        clone_pb.set_message(format!("Setting up SSH connection to: {}", url));

        let ssh_key_path = args.ssh_key.as_ref().map(PathBuf::from).unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
            PathBuf::from(home).join(".ssh/id_rsa")
        });

        if !ssh_key_path.exists() {
            clone_pb.finish_with_message("✗ SSH key not found");
            return Err(anyhow::anyhow!(
                "SSH key not found at {}.\n\
                Please ensure your SSH key exists or specify a different path with --ssh-key",
                ssh_key_path.display()
            ));
        }

        // First try without passphrase
        clone_pb.set_message(format!("Attempting SSH connection to: {}", url));
        let passphrase = args.ssh_passphrase.clone();
        callbacks.credentials(move |_url, _username_from_url, _allowed_types| {
            git2::Cred::ssh_key(
                _username_from_url.unwrap_or("git"),
                None,
                &ssh_key_path,
                passphrase.as_deref(),
            )
        });
        fetch_options.remote_callbacks(callbacks);
        builder.fetch_options(fetch_options);

        let clone_result = builder.clone(url, path);

        if let Err(e) = &clone_result {
            if e.class() == git2::ErrorClass::Ssh
                && e.message().contains("Unable to extract public key")
                && args.ssh_passphrase.is_none()
            {
                // Try again with passphrase
                let passphrase = prompt_passphrase(&clone_pb)?;

                clone_pb.set_message(format!("Retrying SSH connection to: {}", url));
                let mut callbacks = git2::RemoteCallbacks::new();
                let ssh_key_path = args.ssh_key.as_ref().map(PathBuf::from).unwrap_or_else(|| {
                    let home = std::env::var("HOME").unwrap_or_else(|_| "~".to_string());
                    PathBuf::from(home).join(".ssh/id_rsa")
                });

                callbacks.credentials(move |_url, _username_from_url, _allowed_types| {
                    git2::Cred::ssh_key(
                        _username_from_url.unwrap_or("git"),
                        None,
                        &ssh_key_path,
                        Some(&passphrase),
                    )
                });

                let mut fetch_options = git2::FetchOptions::new();
                fetch_options.remote_callbacks(callbacks);
                builder.fetch_options(fetch_options);

                builder
                    .clone(url, path)
                    .map_err(|e| handle_auth_error(url, &e))
            } else {
                clone_result.map_err(|e| handle_auth_error(url, &e))
            }
        } else {
            clone_result.map_err(|e| handle_auth_error(url, &e))
        }
    } else {
        clone_pb.finish_with_message("✗ Invalid URL format");
        Err(anyhow::anyhow!(
            "Invalid repository URL format: {}\n\
            URL must start with 'https://' or 'git@'",
            url
        ))
    };

    // Update progress bar based on result
    match &result {
        Ok(_) => {
            if url.starts_with("git@") {
                clone_pb.finish_with_message(format!(
                    "✓ SSH connection established and repository cloned in {:.1}s",
                    clone_pb.elapsed().as_secs_f64()
                ));
            } else {
                clone_pb.finish_with_message(format!(
                    "✓ Repository cloned in {:.1}s",
                    clone_pb.elapsed().as_secs_f64()
                ));
            }
        }
        Err(_) => {
            clone_pb.finish_with_message("✗ Failed to clone repository");
        }
    }

    result
}

fn process_repository(
    url: &str,
    output_dir: &str,
    stats: Arc<Mutex<ProcessingStats>>,
    args: &Args,
    copy_mode: bool,
    allow_commit: bool,
    multi_progress: Arc<MultiProgress>,
) -> Result<()> {
    let clone_start = Instant::now();

    // Determine the repository directory
    let repo_dir = if url == "." {
        // Use current directory
        std::env::current_dir()?
    } else if let Some(path) = &args.at {
        PathBuf::from(path)
    } else if args.open_cursor {
        // Use cache directory for cursor mode if no specific path provided
        let cache_dir = dirs::cache_dir()
            .ok_or_else(|| anyhow::anyhow!("Could not determine cache directory"))?
            .join("repod");
        fs::create_dir_all(&cache_dir)?;
        cache_dir.join(extract_repo_name(url))
    } else {
        TempDir::new()?.into_path()
    };

    // Only clone if it's a remote repository
    if url != "." {
        // If directory exists and is not empty, remove it first
        if repo_dir.exists() {
            if repo_dir.read_dir()?.next().is_some() {
                println!(
                    "Directory exists and is not empty, removing: {}",
                    repo_dir.display()
                );
                fs::remove_dir_all(&repo_dir)?;
            }
        }

        let _repo = clone_repository(url, &repo_dir, args, &multi_progress)
            .with_context(|| format!("Failed to access repository: {}", url))?;

        {
            let mut stats_guard = stats.lock();
            stats_guard.repo_count += 1;
            stats_guard.clone_time += clone_start.elapsed().as_secs_f64();
        }
    }

    // If commit-only mode is enabled, skip scanning/output and just run commit flow
    if allow_commit {
        // On first use of commit features, ensure GEMINI_API_KEY is configured
        ensure_gemini_api_key_interactive()?;
        if args.multi_commit && args.commit {
            print_warn("Both --commit and --multi-commit provided; choose one. Skipping commit.");
        } else if args.multi_commit {
            commit_with_ai_multi(
                &repo_dir,
                &multi_progress,
                args.branch.as_deref(),
                args.push,
            )?;
        } else if args.commit {
            commit_with_ai_single(
                &repo_dir,
                &multi_progress,
                args.branch.as_deref(),
                args.push,
            )?;
        }
        return Ok(());
    }

    let process_start = Instant::now();

    // Create tokenizer once
    let tokenizer = Arc::new(o200k_base().unwrap());

    // First, check for README file in root
    let scan_pb = multi_progress.add(ProgressBar::new_spinner());
    scan_pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.blue} {msg}")
            .unwrap(),
    );
    scan_pb.enable_steady_tick(std::time::Duration::from_millis(100));
    scan_pb.set_message("Scanning repository structure...");

    let mut readme_content: Option<FileContent> = None;
    // Build only-set matcher once for this repo
    let only_set = build_only_globset(&args.only, &args.only_dirs);

    for readme_name in [
        "README.md",
        "README.txt",
        "README",
        "Readme.md",
        "readme.md",
    ] {
        let readme_path = repo_dir.join(readme_name);
        if readme_path.exists() && readme_path.is_file() {
            // Respect only globs (including only-dir)
            if let Some(ref set) = only_set {
                if !set.is_match(readme_name) {
                    continue;
                }
            }

            if let Ok(content) = read_file_content(&readme_path) {
                let token_count = tokenizer.encode_ordinary(&content).len();
                let metadata_block = build_metadata_block(readme_name);
                let metadata_token_count = tokenizer.encode_ordinary(&metadata_block).len();
                readme_content = Some(FileContent {
                    path: readme_name.to_string(),
                    content,
                    token_count,
                    metadata_token_count,
                });
                break;
            }
        }
    }

    // Build combined exclude matcher (built‑in + user‑supplied)
    let exclude_set = build_exclude_globset(EXCLUDED_PATTERNS, &args.exclude);

    // Build the walker with ignore support
    let mut walker_builder = WalkBuilder::new(&repo_dir);

    // Configure the walker
    // For cloned repos, we disable git-specific ignores to ensure consistent behavior
    // regardless of how the repo was obtained (cloned vs downloaded)
    let is_cloned_repo = url != ".";

    walker_builder
        .hidden(false) // We'll handle hidden files with our own logic
        .git_ignore(true) // Always respect .gitignore files in the repo
        .git_global(!is_cloned_repo) // Only respect global gitignore for local repos
        .git_exclude(!is_cloned_repo) // Only respect .git/info/exclude for local repos
        .ignore(true) // Respect .ignore files
        .parents(!is_cloned_repo); // Only respect parent ignore files for local repos

    // Count total files first for progress bar
    let total_files: usize = walker_builder
        .build()
        .filter_map(Result::ok)
        .filter(|entry| {
            let path = entry.path();
            let rel = normalize_rel_path(path, &repo_dir);

            // Check our built-in + user exclusions (repo-relative)
            let is_excluded = exclude_set
                .as_ref()
                .map(|set| set.is_match(&rel))
                .unwrap_or(false);

            // Check if it's a hidden file/folder (starts with .)
            // Only check path components RELATIVE to the repo_dir to avoid issues with temp directories
            let is_hidden = if let Ok(relative_path) = path.strip_prefix(&repo_dir) {
                relative_path.components().any(|component| {
                    if let std::path::Component::Normal(name) = component {
                        name.to_string_lossy().starts_with('.')
                    } else {
                        false
                    }
                })
            } else {
                // If we can't get relative path, check the full path (fallback)
                path.file_name()
                    .map(|name| name.to_string_lossy().starts_with('.'))
                    .unwrap_or(false)
            };

            let is_file = entry.file_type().map(|ft| ft.is_file()).unwrap_or(false);

            if !(is_file && !is_excluded && !is_hidden) {
                return false;
            }
            if let Some(ref set) = only_set {
                if !set.is_match(&rel) {
                    return false;
                }
            }

            true
        })
        .count();

    scan_pb.finish_with_message(format!("Found {} files", total_files));

    // Process files progress bar
    let process_pb = multi_progress.add(ProgressBar::new(total_files as u64));
    process_pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} files ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );
    process_pb.enable_steady_tick(std::time::Duration::from_millis(100));

    // Collect and process other files in parallel
    let files: Vec<_> = walker_builder
        .build()
        .filter_map(Result::ok)
        .filter(|entry| {
            let path = entry.path();
            let rel = normalize_rel_path(path, &repo_dir);

            // Check our built-in + user exclusions (repo-relative)
            let is_excluded = exclude_set
                .as_ref()
                .map(|set| set.is_match(&rel))
                .unwrap_or(false);

            // Check if it's a hidden file/folder (starts with .)
            // Only check path components RELATIVE to the repo_dir to avoid issues with temp directories
            let is_hidden = if let Ok(relative_path) = path.strip_prefix(&repo_dir) {
                relative_path.components().any(|component| {
                    if let std::path::Component::Normal(name) = component {
                        name.to_string_lossy().starts_with('.')
                    } else {
                        false
                    }
                })
            } else {
                // If we can't get relative path, check the full path (fallback)
                path.file_name()
                    .map(|name| name.to_string_lossy().starts_with('.'))
                    .unwrap_or(false)
            };

            let ok = entry.file_type().map(|ft| ft.is_file()).unwrap_or(false)
                && !is_excluded
                && !is_hidden;
            if !ok {
                return false;
            }
            if let Some(ref set) = only_set {
                if !set.is_match(&rel) {
                    return false;
                }
            }
            true
        })
        .par_bridge()
        .progress_with(process_pb.clone())
        .filter_map(|entry: DirEntry| {
            let path = entry.path();
            // Skip if this is the README we already processed
            if let Some(ref readme) = readme_content {
                if path.file_name().and_then(|n| n.to_str()) == Some(&readme.path) {
                    return None;
                }
            }

            let should_process = should_process_file(
                path,
                &repo_dir,
                if args.repo_types.is_empty() {
                    None
                } else {
                    Some(&args.repo_types)
                },
                only_set.as_ref(),
                exclude_set.as_ref(),
            );
            let is_binary = matches!(is_binary_file(path), Ok(true));

            if !should_process || is_binary {
                if is_binary {
                    // Increment binary skipped counter if is_binary is true
                    stats.lock().binary_files_skipped += 1;
                }
                return None;
            }

            read_file_content(path).ok().map(|content| {
                let relative_path = path.strip_prefix(&repo_dir).unwrap().display().to_string();
                let token_count = tokenizer.encode_ordinary(&content).len();
                let metadata_block = build_metadata_block(&relative_path);
                let metadata_token_count = tokenizer.encode_ordinary(&metadata_block).len();
                FileContent {
                    path: relative_path,
                    content,
                    token_count,
                    metadata_token_count,
                }
            })
        })
        .collect();

    process_pb.finish_with_message(format!("Processed {} files", files.len()));

    // Prepare directory tree output for later writing and token accounting
    let tree = DirectoryTree::build(&repo_dir, exclude_set.as_ref(), &args.only, &args.only_dirs)?;
    let directory_block = format!(
        "<directory_structure>\n{}\n</directory_structure>\n\n",
        tree.format()
    );
    let directory_token_count = tokenizer.encode_ordinary(&directory_block).len();

    let file_token_total: usize = files.iter().map(|f| f.token_count).sum();
    let file_metadata_total: usize = files.iter().map(|f| f.metadata_token_count).sum();
    let readme_token_total = readme_content.as_ref().map(|f| f.token_count).unwrap_or(0);
    let readme_metadata_total = readme_content
        .as_ref()
        .map(|f| f.metadata_token_count)
        .unwrap_or(0);
    let file_count_including_readme = files.len() + (readme_content.is_some() as usize);
    let spacing_token_unit = tokenizer.encode_ordinary("\n\n").len();
    let spacing_token_total = spacing_token_unit * file_count_including_readme;

    // Update stats
    {
        let mut stats_guard = stats.lock();
        stats_guard.total_files += files.len() + (readme_content.is_some() as usize);

        let repo_token_total = file_token_total
            + file_metadata_total
            + directory_token_count
            + readme_token_total
            + readme_metadata_total
            + spacing_token_total;
        stats_guard.total_tokens += repo_token_total;

        stats_guard.processing_time += process_start.elapsed().as_secs_f64();
    }

    // Write progress
    let write_pb = multi_progress.add(ProgressBar::new_spinner());
    write_pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap(),
    );
    write_pb.enable_steady_tick(std::time::Duration::from_millis(100));
    write_pb.set_message("Writing output");

    // Create output content
    let mut output_buffer = Vec::new();

    // First, write the directory tree
    output_buffer.write_all(directory_block.as_bytes())?;

    // Write README first if it exists
    if let Some(readme) = readme_content {
        process_files_batch(&[readme], &mut output_buffer)?;
    }

    // Write remaining files in chunks
    for chunk in files.chunks(CHUNK_SIZE) {
        process_files_batch(chunk, &mut output_buffer)?;
    }

    // Handle output based on mode
    if copy_mode {
        // Copy to clipboard
        let content = String::from_utf8(output_buffer)?;
        let mut ctx = ClipboardContext::new()
            .map_err(|e| anyhow::anyhow!("Failed to access clipboard: {}", e))?;
        ctx.set_contents(content)
            .map_err(|e| anyhow::anyhow!("Failed to copy to clipboard: {}", e))?;
        println!("Content copied to clipboard");
    } else {
        // Write to file
        let output_file_name = if args.open_cursor {
            // In cursor mode, write to the repo root
            let timestamp = Local::now().format("%Y%m%d_%H%M%S");
            repo_dir.join(format!("screenpipe_{}.txt", timestamp))
        } else {
            let timestamp = Local::now().format("%Y%m%d_%H%M%S");
            let repo_name = if url == "." {
                repo_dir.file_name().unwrap().to_string_lossy().to_string()
            } else {
                extract_repo_name(url)
            };
            PathBuf::from(format!("{}/{}_{}.txt", output_dir, repo_name, timestamp))
        };
        let mut file = File::create(&output_file_name)?;
        file.write_all(&output_buffer)?;
    }

    write_pb.finish_with_message("Finished writing output");

    // Make sure all progress bars are properly cleaned up
    drop(scan_pb);
    drop(process_pb);
    drop(write_pb);
    multi_progress.clear()?;

    // If cursor mode is enabled, run the cursor command
    if args.open_cursor {
        let cursor_cmd = format!("cursor {}", repo_dir.display());
        if let Err(e) = std::process::Command::new("sh")
            .arg("-c")
            .arg(&cursor_cmd)
            .spawn()
        {
            println!("Failed to open Cursor: {}", e);
        }
    }

    Ok(())
}

// -------------------- Commit support --------------------

// (old commit_with_ai_message/commit_with_ai_choice removed)

fn commit_with_ai_single(
    repo_dir: &Path,
    multi_progress: &MultiProgress,
    branch_spec: Option<&str>,
    do_push: bool,
) -> Result<()> {
    if !repo_dir.join(".git").exists() {
        print_warn(&format!("Not a git repository: {}", repo_dir.display()));
        return Ok(());
    }
    let current_branch = ensure_on_target_branch(repo_dir, branch_spec, multi_progress)?;
    print_title(&format!("AI Commit (Single) — branch: {}", current_branch));
    let status_porcelain = run_in_repo(repo_dir, &["git", "status", "--porcelain"])?;
    if status_porcelain.trim().is_empty() {
        print_info("No changes detected. Nothing to commit.");
        return Ok(());
    }

    let pb = multi_progress.add(ProgressBar::new_spinner());
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg} [{elapsed_precise}]")
            .unwrap(),
    );
    pb.enable_steady_tick(std::time::Duration::from_millis(100));
    pb.set_message("Generating single-commit proposal...");
    let diff_base = diff_base_ref(repo_dir);
    let name_status = run_in_repo(repo_dir, &["git", "diff", "--name-status", diff_base])?;
    let shortstat = run_in_repo(repo_dir, &["git", "diff", "--shortstat", diff_base])?;
    let numstat = run_in_repo(repo_dir, &["git", "diff", "--numstat", diff_base])?;
    let changes_box = build_changes_summary_box(&numstat, &shortstat, 50);
    print_boxed("Changes", &changes_box);
    let diff_sample = truncate(
        &run_in_repo(repo_dir, &["git", "diff", "-U3", diff_base])?,
        20_000,
    );
    let prompt = build_commit_prompt_multiline(&name_status, &shortstat, &diff_sample);
    let msg = match generate_commit_message_via_gemini(&prompt) {
        Ok(m) => m,
        Err(_) => fallback_commit_message_multiline(&name_status, &shortstat),
    };
    pb.finish_with_message(format!(
        "{}",
        "Single-commit proposal ready".to_string().green().bold()
    ));

    // Show message and confirm
    print_boxed("Proposed Commit", &msg);
    if !prompt_yes_no_keypress("› Commit with this message? [y/N] ")? {
        print_info("Commit canceled.");
        return Ok(());
    }

    // Stage and commit
    run_in_repo(repo_dir, &["git", "add", "-A"])?;
    if let Some((subject, body)) = split_subject_body(&msg) {
        if body.trim().is_empty() {
            run_in_repo(repo_dir, &["git", "commit", "-m", subject.trim()])?;
        } else {
            run_in_repo(
                repo_dir,
                &["git", "commit", "-m", subject.trim(), "-m", body.trim()],
            )?;
        }
    } else {
        run_in_repo(repo_dir, &["git", "commit", "-m", msg.trim()])?;
    }
    print_success(&format!("Committed to {}.", current_branch));

    if do_push {
        try_push(repo_dir, &current_branch)?;
    }

    let leftovers = list_changed_files_vs_head(repo_dir)?;
    if !leftovers.is_empty() {
        print_warn(&format!("Leftover uncommitted files: {}", leftovers.len()));
        for f in &leftovers {
            println!("  • {}", f);
        }
        if prompt_yes_no_keypress("› Generate AI commit for leftovers? [y/N] ")? {
            commit_files_with_ai(repo_dir, &leftovers, multi_progress)?;
            print_success("Leftover files committed.");
        }
    }
    Ok(())
}

fn commit_with_ai_multi(
    repo_dir: &Path,
    multi_progress: &MultiProgress,
    branch_spec: Option<&str>,
    do_push: bool,
) -> Result<()> {
    if !repo_dir.join(".git").exists() {
        print_warn(&format!("Not a git repository: {}", repo_dir.display()));
        return Ok(());
    }
    let current_branch = ensure_on_target_branch(repo_dir, branch_spec, multi_progress)?;
    print_title(&format!("AI Commit (Multi) — branch: {}", current_branch));
    let status_porcelain = run_in_repo(repo_dir, &["git", "status", "--porcelain"])?;
    if status_porcelain.trim().is_empty() {
        print_info("No changes detected. Nothing to commit.");
        return Ok(());
    }

    let pb = multi_progress.add(ProgressBar::new_spinner());
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg} [{elapsed_precise}]")
            .unwrap(),
    );
    pb.enable_steady_tick(std::time::Duration::from_millis(100));
    pb.set_message("Analyzing multi-commit plan...");
    let (commits, leftovers) = plan_multi_commits(repo_dir, multi_progress)?;
    let diff_base = diff_base_ref(repo_dir);
    let shortstat = run_in_repo(repo_dir, &["git", "diff", "--shortstat", diff_base])?;
    let numstat = run_in_repo(repo_dir, &["git", "diff", "--numstat", diff_base])?;
    let changes_box = build_changes_summary_box(&numstat, &shortstat, 50);
    print_boxed("Changes", &changes_box);
    pb.finish_with_message(format!(
        "{}",
        "Multi-commit analysis complete".to_string().green().bold()
    ));

    println!("Proposed multi-commit plan:\n");
    for (i, c) in commits.iter().enumerate() {
        println!("{}. {}", i + 1, c.title);
        if let Some(body) = &c.body {
            if !body.trim().is_empty() {
                println!("\n{}\n", body.trim());
            }
        }
        println!("Files ({}):", c.files.len());
        for f in &c.files {
            println!("  - {}", f);
        }
        println!("");

        // Per-commit change summary (shortstat + numstat scoped to these files)
        let mut shortstat_args = vec![
            "git".to_string(),
            "diff".to_string(),
            "--shortstat".to_string(),
            diff_base.to_string(),
            "--".to_string(),
        ];
        let mut numstat_args = vec![
            "git".to_string(),
            "diff".to_string(),
            "--numstat".to_string(),
            diff_base.to_string(),
            "--".to_string(),
        ];
        for f in &c.files {
            shortstat_args.push(f.clone());
            numstat_args.push(f.clone());
        }
        if let Ok(shortstat_scoped) = run_in_repo_strings(repo_dir, shortstat_args) {
            if let Ok(numstat_scoped) = run_in_repo_strings(repo_dir, numstat_args) {
                let box_text = build_changes_summary_box(&numstat_scoped, &shortstat_scoped, 50);
                if !box_text.trim().is_empty() {
                    print_boxed("Changes", &box_text);
                }
            }
        }
    }
    if !leftovers.is_empty() {
        print_warn(&format!(
            "Leftover files not in any commit: {}",
            leftovers.len()
        ));
        for f in &leftovers {
            println!("  • {}", f);
        }
        println!("");
    }
    // Confirm and apply each commit individually
    for (i, c) in commits.iter().enumerate() {
        println!("Apply commit {}/{}: {}", i + 1, commits.len(), c.title);
        if let Some(body) = &c.body {
            if !body.trim().is_empty() {
                println!("\n{}\n", body.trim());
            }
        }
        println!("Files ({}):", c.files.len());
        for f in &c.files {
            println!("  - {}", f);
        }
        if prompt_yes_no_keypress("Commit this change? [y/N] ")? {
            let mut add_args = vec![
                "git".to_string(),
                "add".to_string(),
                "-A".to_string(),
                "--".to_string(),
            ];
            for f in &c.files {
                add_args.push(f.clone());
            }
            run_in_repo_strings(repo_dir, add_args)?;

            let subject = c.title.trim().to_string();
            let body = c.body.as_deref().unwrap_or("").trim().to_string();
            if body.is_empty() {
                run_in_repo(repo_dir, &["git", "commit", "-m", &subject])?;
            } else {
                run_in_repo(repo_dir, &["git", "commit", "-m", &subject, "-m", &body])?;
            }
        } else {
            println!("Skipped.");
        }
    }

    let post_leftovers = list_changed_files_vs_head(repo_dir)?;
    if !post_leftovers.is_empty() {
        print_warn(&format!(
            "Leftover uncommitted files: {}",
            post_leftovers.len()
        ));
        for f in &post_leftovers {
            println!("  • {}", f);
        }
        if prompt_yes_no_keypress("› Generate AI commit for leftovers? [y/N] ")? {
            commit_files_with_ai(repo_dir, &post_leftovers, multi_progress)?;
            print_success("Leftover files committed.");
        }
    }
    if do_push {
        try_push(repo_dir, &current_branch)?;
    }
    print_success("Multi-commit completed.");
    Ok(())
}

fn run_in_repo(repo_dir: &Path, args: &[&str]) -> Result<String> {
    let (cmd, rest) = args
        .split_first()
        .ok_or_else(|| anyhow::anyhow!("empty command"))?;
    let output = Command::new(cmd)
        .args(rest)
        .current_dir(repo_dir)
        .output()
        .with_context(|| format!("failed to run {:?}", args))?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        Err(anyhow::anyhow!(
            "command {:?} failed: {}",
            args,
            stderr.trim()
        ))
    }
}

fn git_has_head(repo_dir: &Path) -> bool {
    Command::new("git")
        .args(["rev-parse", "--verify", "HEAD"])
        .current_dir(repo_dir)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn diff_base_ref(repo_dir: &Path) -> &'static str {
    if git_has_head(repo_dir) {
        "HEAD"
    } else {
        EMPTY_TREE_HASH
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }

    let mut end = max.min(s.len());
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }

    let prefix = &s[..end];
    if prefix.len() == s.len() {
        s.to_string()
    } else {
        format!("{}\n…[truncated]", prefix)
    }
}

fn prompt_yes_no_keypress(prompt: &str) -> Result<bool> {
    use std::io::Write;
    print!("{}", prompt);
    std::io::stdout().flush().ok();
    terminal::enable_raw_mode().map_err(|e| anyhow::anyhow!("failed to enable raw mode: {}", e))?;
    let res = loop {
        match read() {
            Ok(Event::Key(key)) => match key.code {
                KeyCode::Char(c) => {
                    let cl = c.to_ascii_lowercase();
                    match cl {
                        'y' => {
                            print!("{}\n", c);
                            std::io::stdout().flush().ok();
                            break Ok(true);
                        }
                        'n' => {
                            print!("{}\n", c);
                            std::io::stdout().flush().ok();
                            break Ok(false);
                        }
                        _ => {}
                    }
                }
                KeyCode::Esc => {
                    print!("\n");
                    std::io::stdout().flush().ok();
                    break Ok(false);
                }
                _ => {}
            },
            Ok(_) => {}
            Err(e) => break Err(anyhow::anyhow!("failed to read key: {}", e)),
        }
    };
    terminal::disable_raw_mode().ok();
    res
}

fn prompt_choice_keypress(prompt: &str, allowed: &[char]) -> Result<char> {
    use std::io::Write;
    print!("{}", prompt);
    std::io::stdout().flush().ok();
    terminal::enable_raw_mode().map_err(|e| anyhow::anyhow!("failed to enable raw mode: {}", e))?;
    let res = loop {
        match read() {
            Ok(Event::Key(key)) => match key.code {
                KeyCode::Char(c) => {
                    let cl = c.to_ascii_lowercase();
                    if allowed.contains(&cl) {
                        // echo selection and newline for feedback
                        print!("{}\n", c);
                        std::io::stdout().flush().ok();
                        break Ok(cl);
                    }
                }
                KeyCode::Esc => break Ok('c'),
                KeyCode::Enter => { /* ignore */ }
                _ => {}
            },
            Ok(_) => {}
            Err(e) => break Err(anyhow::anyhow!("failed to read key: {}", e)),
        }
    };
    terminal::disable_raw_mode().ok();
    res
}

fn split_subject_body(msg: &str) -> Option<(String, String)> {
    let mut lines = msg.lines();
    let subject = lines.next()?.to_string();
    let rest: String = lines.collect::<Vec<&str>>().join("\n");
    Some((subject, rest))
}

fn read_line_prompt(prompt: &str) -> Result<String> {
    use std::io::{self, Write};
    print!("{}", prompt);
    io::stdout().flush().ok();
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .map_err(|e| anyhow::anyhow!("failed to read input: {}", e))?;
    Ok(input.trim().to_string())
}

fn build_commit_prompt_multiline(name_status: &str, shortstat: &str, diff_sample: &str) -> String {
    format!(
        "You write excellent Conventional Commits. Generate a concise, multi-line commit message:\n\
        - First line: <type>(optional-scope): <summary> (<=72 chars, no trailing period)\n\
        - Blank line\n\
        - Body: 3-6 bullets summarizing key changes and rationale; wrap to ~72 chars\n\
        - Include 'BREAKING CHANGE:' line if applicable\n\
        Prefer specific wording over generic 'update' or 'changes'.\n\
        Changed files (name-status):\n\
        {}\n\
        Summary: {}\n\
        Diff sample (truncated):\n\
        {}\n\
        Output ONLY the commit message text.",
        name_status.trim(),
        shortstat.trim(),
        diff_sample.trim()
    )
}

fn fallback_commit_message_multiline(name_status: &str, shortstat: &str) -> String {
    // Simple heuristic fallback if API not available (multi-line)
    let files: Vec<&str> = name_status
        .lines()
        .take(5)
        .map(|l| l.split_whitespace().last().unwrap_or(l))
        .collect();
    let files_str = files.join(", ");
    let stat = shortstat.trim();
    let subject = if files_str.is_empty() {
        "chore: update files".to_string()
    } else {
        truncate(&format!("chore: update {}", files_str), 72)
    };
    let body = format!(
        "\n\n- Update files\n- Summary: {}",
        if stat.is_empty() { "n/a" } else { stat }
    );
    format!("{}{}", subject, body)
}

#[derive(Serialize)]
struct GeminiRequest<'a> {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GeminiTool<'a>>>,
    #[serde(rename = "toolConfig", skip_serializing_if = "Option::is_none")]
    tool_config: Option<GeminiToolConfig<'a>>,
}

#[derive(Serialize, Clone)]
struct GeminiContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    parts: Vec<GeminiPart>,
}

#[derive(Serialize, Clone)]
struct GeminiPart {
    text: String,
}

#[derive(Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
}

#[derive(Deserialize)]
struct GeminiCandidate {
    content: Option<GeminiGeneratedContent>,
}

#[derive(Deserialize)]
struct GeminiGeneratedContent {
    parts: Option<Vec<GeminiGeneratedPart>>,
}

#[derive(Deserialize)]
struct GeminiGeneratedPart {
    text: Option<String>,
    #[serde(rename = "functionCall")]
    function_call: Option<GeminiFunctionCall>,
}

#[derive(Deserialize)]
struct GeminiFunctionCall {
    name: String,
    #[serde(default)]
    args: serde_json::Value,
}

#[derive(Serialize)]
struct GeminiTool<'a> {
    #[serde(rename = "functionDeclarations")]
    function_declarations: Vec<GeminiFunctionDeclaration<'a>>,
}

#[derive(Serialize)]
struct GeminiFunctionDeclaration<'a> {
    name: &'a str,
    description: &'a str,
    parameters: serde_json::Value,
}

#[derive(Serialize)]
struct GeminiToolConfig<'a> {
    #[serde(rename = "functionCallingConfig")]
    function_calling_config: GeminiFunctionCallingConfig<'a>,
}

#[derive(Serialize)]
struct GeminiFunctionCallingConfig<'a> {
    mode: &'a str,
    #[serde(
        rename = "allowedFunctionNames",
        skip_serializing_if = "Option::is_none"
    )]
    allowed_function_names: Option<Vec<&'a str>>,
}

fn generate_commit_message_via_gemini(prompt: &str) -> Result<String> {
    let api_key =
        std::env::var("GEMINI_API_KEY").map_err(|_| anyhow::anyhow!("GEMINI_API_KEY not set"))?;
    let model = "gemini-3-flash-preview";
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        model, api_key
    );

    let req = GeminiRequest {
        contents: vec![GeminiContent {
            role: Some("user".to_string()),
            parts: vec![GeminiPart {
                text: prompt.to_string(),
            }],
        }],
        tools: None,
        tool_config: None,
    };
    let resp: GeminiResponse = ureq::post(&url)
        .set("Content-Type", "application/json")
        .send_json(serde_json::to_value(&req)?)
        .map_err(|e| anyhow::anyhow!("Gemini request failed: {}", e))?
        .into_json()
        .map_err(|e| anyhow::anyhow!("invalid Gemini JSON: {}", e))?;

    let text = resp
        .candidates
        .and_then(|mut v| v.pop())
        .and_then(|c| c.content)
        .and_then(|c| c.parts)
        .and_then(|mut parts| parts.pop())
        .and_then(|p| p.text)
        .unwrap_or_default()
        .trim()
        .to_string();
    if text.is_empty() {
        anyhow::bail!("empty response from model")
    } else {
        Ok(text)
    }
}

// -------- Multi-commit planning --------

#[derive(Debug, Deserialize)]
struct CommitPlanResponse {
    commits: Vec<CommitPlan>,
}

#[derive(Debug, Deserialize)]
struct CommitPlan {
    title: String,
    body: Option<String>,
    files: Vec<String>,
}

fn plan_multi_commits(
    repo_dir: &Path,
    _multi_progress: &MultiProgress,
) -> Result<(Vec<CommitPlan>, Vec<String>)> {
    // Ensure repo and changes
    if !repo_dir.join(".git").exists() {
        anyhow::bail!("Not a git repository: {}", repo_dir.display());
    }
    let status_porcelain = run_in_repo(repo_dir, &["git", "status", "--porcelain"])?;
    if status_porcelain.trim().is_empty() {
        anyhow::bail!("no changes to commit");
    }

    // Gather change context
    let diff_base = diff_base_ref(repo_dir);
    let name_status = run_in_repo(repo_dir, &["git", "diff", "--name-status", diff_base])?;
    let numstat = run_in_repo(repo_dir, &["git", "diff", "--numstat", diff_base])?;
    let shortstat = run_in_repo(repo_dir, &["git", "diff", "--shortstat", diff_base])?;
    let diff_sample = truncate(
        &run_in_repo(repo_dir, &["git", "diff", "-U3", diff_base])?,
        40_000,
    );

    let plan_prompt = build_multi_commit_prompt(&name_status, &numstat, &shortstat, &diff_sample);
    let plan = match generate_commit_plan_via_gemini(&plan_prompt) {
        Ok(p) => p,
        Err(e) => {
            return Err(anyhow::anyhow!("AI planning failed: {}", e));
        }
    };

    // Collect actually changed files for validation
    let changed_files: Vec<String> = name_status
        .lines()
        .filter_map(|l| l.split_whitespace().nth(1))
        .map(|s| s.to_string())
        .collect();

    // Validate and normalize plan
    let mut normalized: Vec<CommitPlan> = Vec::new();
    for mut c in plan.commits {
        c.files.retain(|f| changed_files.iter().any(|cf| cf == f));
        if !c.title.trim().is_empty() && !c.files.is_empty() {
            normalized.push(c);
        }
    }

    if normalized.is_empty() {
        anyhow::bail!("AI did not propose any valid commits");
    }

    // Determine leftovers
    let mut included = std::collections::HashSet::new();
    for c in &normalized {
        for f in &c.files {
            included.insert(f.clone());
        }
    }
    let leftovers: Vec<String> = changed_files
        .into_iter()
        .filter(|f| !included.contains(f))
        .collect();

    Ok((normalized, leftovers))
}

// (old do_commits removed)

fn build_multi_commit_prompt(
    name_status: &str,
    numstat: &str,
    shortstat: &str,
    diff_sample: &str,
) -> String {
    format!(
        "Analyze the following changes and propose a set of logical commits.\n\
        Output STRICT JSON with this schema: {{\"commits\":[{{\"title\":string,\"body\":string,\"files\":[string]}}]}}.\n\
        Rules:\n\
        - Group changes by intent/scope so each commit is meaningful.\n\
        - Use Conventional Commit titles (<=72 chars).\n\
        - Body should briefly explain rationale and key changes (optional).\n\
        - Assign each changed file to at most one commit.\n\
        Changed files (name-status):\n{}\n\
        Per-file stats (numstat):\n{}\n\
        Summary: {}\n\
        Diff sample (truncated):\n{}\n\
        JSON only.",
        name_status.trim(), numstat.trim(), shortstat.trim(), diff_sample.trim()
    )
}

fn generate_commit_plan_via_gemini(prompt: &str) -> Result<CommitPlanResponse> {
    let api_key =
        std::env::var("GEMINI_API_KEY").map_err(|_| anyhow::anyhow!("GEMINI_API_KEY not set"))?;
    let model = "gemini-3-flash-preview";
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        model, api_key
    );

    // Declare a function tool for structured multi-commit planning
    let params_schema = serde_json::json!({
        "type": "object",
        "properties": {
            "commits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": { "type": "string" },
                        "body":  { "type": "string" },
                        "files": { "type": "array", "items": { "type": "string" } }
                    },
                    "required": ["title", "files"]
                }
            }
        },
        "required": ["commits"]
    });

    let req = GeminiRequest {
        contents: vec![GeminiContent {
            role: Some("user".to_string()),
            parts: vec![GeminiPart {
                text: prompt.to_string(),
            }],
        }],
        tools: Some(vec![GeminiTool {
            function_declarations: vec![GeminiFunctionDeclaration {
                name: "propose_commit_plan",
                description:
                    "Propose a logical multi-commit plan for the provided repository changes.",
                parameters: params_schema,
            }],
        }]),
        tool_config: Some(GeminiToolConfig {
            function_calling_config: GeminiFunctionCallingConfig {
                mode: "ANY",
                allowed_function_names: Some(vec!["propose_commit_plan"]),
            },
        }),
    };

    let resp: GeminiResponse = ureq::post(&url)
        .set("Content-Type", "application/json")
        .send_json(serde_json::to_value(&req)?)
        .map_err(|e| anyhow::anyhow!("Gemini request failed: {}", e))?
        .into_json()
        .map_err(|e| anyhow::anyhow!("invalid Gemini JSON: {}", e))?;

    // Prefer tool-calling path: extract function call arguments
    let candidates = resp.candidates.unwrap_or_default();
    for cand in &candidates {
        if let Some(content) = &cand.content {
            if let Some(parts) = &content.parts {
                for part in parts {
                    if let Some(fc) = &part.function_call {
                        // Accept only our declared function
                        if fc.name == "propose_commit_plan" {
                            // args might be a struct or a JSON string – handle both
                            let plan_res: Result<CommitPlanResponse> = match &fc.args {
                                serde_json::Value::String(s) => {
                                    if let Ok(plan) = serde_json::from_str::<CommitPlanResponse>(s)
                                    {
                                        Ok(plan)
                                    } else if let Ok(commits) =
                                        serde_json::from_str::<Vec<CommitPlan>>(s)
                                    {
                                        Ok(CommitPlanResponse { commits })
                                    } else {
                                        Err(anyhow::anyhow!(
                                            "functionCall args string not valid plan JSON"
                                        ))
                                    }
                                }
                                v => {
                                    if let Ok(plan) =
                                        serde_json::from_value::<CommitPlanResponse>(v.clone())
                                    {
                                        Ok(plan)
                                    } else if let Ok(commits) =
                                        serde_json::from_value::<Vec<CommitPlan>>(v.clone())
                                    {
                                        Ok(CommitPlanResponse { commits })
                                    } else {
                                        Err(anyhow::anyhow!(
                                            "functionCall args not valid plan JSON"
                                        ))
                                    }
                                }
                            };
                            if let Ok(plan) = plan_res {
                                return Ok(plan);
                            }
                        }
                    }
                }
            }
        }
    }

    // Fallback: parse any text output as before (robust JSON extraction)
    let mut last_text: Option<String> = None;
    for cand in candidates {
        if let Some(content) = cand.content {
            if let Some(parts) = content.parts {
                for part in parts {
                    if let Some(t) = part.text {
                        last_text = Some(t);
                    }
                }
            }
        }
    }

    fn extract_json_candidate(s: &str) -> Option<String> {
        let t = s.trim();
        if t.is_empty() {
            return None;
        }
        if let Some(start) = t.find("```") {
            let after = &t[start + 3..];
            let after = after
                .strip_prefix("json")
                .or_else(|| after.strip_prefix("JSON"))
                .unwrap_or(after);
            let after = after.strip_prefix('\n').unwrap_or(after);
            if let Some(end_rel) = after.find("```") {
                let block = &after[..end_rel];
                let block_trim = block.trim();
                if block_trim.starts_with('{') || block_trim.starts_with('[') {
                    return Some(block_trim.to_string());
                }
            }
        }
        let mut depth = 0usize;
        let mut start_idx: Option<usize> = None;
        for (i, ch) in t.char_indices() {
            match ch {
                '{' => {
                    if depth == 0 {
                        start_idx = Some(i);
                    }
                    depth += 1;
                }
                '}' => {
                    if depth > 0 {
                        depth -= 1;
                    }
                    if depth == 0 {
                        if let Some(s0) = start_idx {
                            return Some(t[s0..=i].to_string());
                        }
                    }
                }
                _ => {}
            }
        }
        // Try array scanning
        if let Some(s0) = t.find('[') {
            if let Some(s1) = t.rfind(']') {
                if s1 > s0 {
                    return Some(t[s0..=s1].to_string());
                }
            }
        }
        None
    }

    if let Some(text) = last_text {
        let trimmed = text.trim();
        if let Ok(plan) = serde_json::from_str::<CommitPlanResponse>(trimmed) {
            return Ok(plan);
        }
        if let Some(candidate) = extract_json_candidate(trimmed) {
            if let Ok(plan) = serde_json::from_str::<CommitPlanResponse>(&candidate) {
                return Ok(plan);
            }
            if let Ok(commits) = serde_json::from_str::<Vec<CommitPlan>>(&candidate) {
                return Ok(CommitPlanResponse { commits });
            }
        }
        if let Ok(commits) = serde_json::from_str::<Vec<CommitPlan>>(trimmed) {
            return Ok(CommitPlanResponse { commits });
        }
    }
    anyhow::bail!("no function call found and could not parse text output as JSON")
}

// -------------------- Ask repo (Q&A) --------------------

fn ask_about_repository(
    repo_dir: &Path,
    question: &str,
    args: &Args,
    multi_progress: &MultiProgress,
) -> Result<()> {
    print_title("Ask (Repository)");

    // Build repository dump (tree + selected files)
    let pb = multi_progress.add(ProgressBar::new_spinner());
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg} [{elapsed_precise}]")
            .unwrap(),
    );
    pb.enable_steady_tick(std::time::Duration::from_millis(100));
    pb.set_message("Preparing repository context...");
    let t0 = Instant::now();
    let (dump, stats) = build_repo_dump(repo_dir, args)?;
    pb.finish_with_message(format!(
        "{}",
        "Repository context ready".to_string().green().bold()
    ));
    print_info(&format!(
        "Included files: {} | Context bytes: {}",
        stats.files, stats.bytes
    ));

    if stats.files == 0 {
        print_warn("No files matched the current filters. Aborting --ask.\nHint: Adjust --only/--exclude/--only-dir or choose a different path.");
        return Ok(());
    }

    // Build initial conversation with repo context attached to the first user turn
    let tokenizer = o200k_base().unwrap();
    let instructions = "You are assisting with repository analysis.\nAnswer the user's question based on the repository content.\nBe concise and specific; include filenames when relevant.\nIf unsure, say so briefly.";
    let mut history: Vec<GeminiContent> = Vec::new();
    history.push(GeminiContent {
        role: Some("user".to_string()),
        parts: vec![GeminiPart {
            text: format!(
                "{}\nQuestion:\n{}\nRepository:\n{}",
                instructions,
                question.trim(),
                dump
            ),
        }],
    });

    let initial_tokens = count_tokens_for_gemini(&history, &tokenizer);
    if initial_tokens > 1_000_000 {
        print_warn(&format!(
            "Context too large ({} tokens > 1,000,000). Aborting request.\nHint: Narrow with --only/--exclude or reduce repository size.",
            initial_tokens
        ));
        return Ok(());
    }
    print_info(&format!(
        "Prompt tokens: {} | Prep time: {:.2}s",
        initial_tokens,
        t0.elapsed().as_secs_f64()
    ));

    // Resolve model name from --model flag
    let model_name = match args.model.to_lowercase().as_str() {
        "flash" => "gemini-3-flash-preview",
        "pro" | _ => "gemini-3-pro-preview",
    };

    let mut turn = 1usize;
    loop {
        let token_count = count_tokens_for_gemini(&history, &tokenizer);
        if token_count > 1_000_000 {
            print_warn("Conversation too large for the model. Restart --ask with narrower filters or shorter history.");
            break;
        }

        print_title(&format!("Answer {} (streaming)", turn));
        let mut streamed = true;
        let answer_text = match generate_repo_answer_stream_via_gemini(&history, model_name) {
            Ok(answer_text) => answer_text,
            Err(e) => {
                streamed = false;
                print_warn(&format!(
                    "Streaming failed ({}). Falling back to non-streaming.",
                    e
                ));
                generate_repo_answer_via_gemini(&history, model_name)?
            }
        };
        if !streamed {
            print_boxed("Answer", &answer_text);
        }

        history.push(GeminiContent {
            role: Some("model".to_string()),
            parts: vec![GeminiPart {
                text: answer_text.clone(),
            }],
        });

        if args.copy {
            if let Ok(mut ctx) = ClipboardContext::new() {
                let _ = ctx.set_contents(answer_text.clone());
            }
            print_success("Answer copied to clipboard.");
        }

        let follow_up = match read_line_prompt("› Ask follow-up (Enter to finish, q to quit): ") {
            Ok(s) => s,
            Err(_) => break,
        };
        let follow_up = follow_up.trim();
        if follow_up.is_empty() || follow_up == "q" || follow_up == ":q" {
            break;
        }

        history.push(GeminiContent {
            role: Some("user".to_string()),
            parts: vec![GeminiPart {
                text: format!(
                    "Follow-up question (repository context is unchanged):\n{}",
                    follow_up
                ),
            }],
        });

        turn += 1;
    }
    Ok(())
}

fn count_tokens_for_gemini(contents: &[GeminiContent], tokenizer: &tiktoken_rs::CoreBPE) -> usize {
    let mut combined = String::new();
    for content in contents {
        if let Some(role) = &content.role {
            combined.push_str(role);
            combined.push_str(": ");
        }
        for part in &content.parts {
            combined.push_str(&part.text);
            combined.push('\n');
        }
    }
    tokenizer.encode_with_special_tokens(&combined).len()
}

struct AskStats {
    files: usize,
    bytes: usize,
}

fn build_repo_dump(repo_dir: &Path, args: &Args) -> Result<(String, AskStats)> {
    // Build combined excluded matcher
    let exclude_set = build_exclude_globset(EXCLUDED_PATTERNS, &args.exclude);

    // Build only matcher once
    let only_set = build_only_globset(&args.only, &args.only_dirs);

    // Tree first
    let mut output = String::new();
    let mut files_included = 0usize;
    output.push_str("<directory_structure>\n");
    let tree = DirectoryTree::build(repo_dir, exclude_set.as_ref(), &args.only, &args.only_dirs)?;
    output.push_str(&tree.format());
    output.push_str("\n</directory_structure>\n\n");

    // README first if exists
    let readme_names = [
        "README.md",
        "README.txt",
        "README",
        "Readme.md",
        "readme.md",
    ];
    for readme_name in readme_names {
        let readme_path = repo_dir.join(readme_name);
        if readme_path.exists() && readme_path.is_file() {
            if let Some(ref set) = only_set {
                if !set.is_match(readme_name) {
                    continue;
                }
            }
            if let Ok(content) = read_file_content(&readme_path) {
                output.push_str("<file_info>\n");
                output.push_str(&format!("path: {}\n", readme_name));
                output.push_str(&format!("name: {}\n", readme_name));
                output.push_str("</file_info>\n");
                output.push_str(&content);
                output.push_str("\n\n");
                files_included += 1;
            }
            break;
        }
    }

    // Walk and include other files
    let mut walker_builder = WalkBuilder::new(repo_dir);
    walker_builder
        .hidden(false)
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true)
        .ignore(true)
        .parents(true);

    for result in walker_builder.build().filter_map(Result::ok) {
        let path = result.path();
        if path == repo_dir {
            continue;
        }
        let rel = normalize_rel_path(path, repo_dir);
        // Exclusions
        if exclude_set
            .as_ref()
            .map(|set| set.is_match(&rel))
            .unwrap_or(false)
        {
            continue;
        }
        // Hidden components
        if let Ok(rel) = path.strip_prefix(repo_dir) {
            let hidden = rel.components().any(|c| matches!(c, std::path::Component::Normal(n) if n.to_string_lossy().starts_with('.')));
            if hidden {
                continue;
            }
        }
        let is_file = result.file_type().map(|ft| ft.is_file()).unwrap_or(false);
        if !is_file {
            continue;
        }

        // Respect only globs
        if let Some(ref set) = only_set {
            if !set.is_match(&rel) {
                continue;
            }
        }

        // Respect repo_types
        if !should_process_file(
            path,
            repo_dir,
            if args.repo_types.is_empty() {
                None
            } else {
                Some(&args.repo_types)
            },
            only_set.as_ref(),
            exclude_set.as_ref(),
        ) {
            continue;
        }
        if matches!(is_binary_file(path), Ok(true)) {
            continue;
        }

        if let Ok(content) = read_file_content(path) {
            let rel = path.strip_prefix(repo_dir).unwrap().display().to_string();
            output.push_str("<file_info>\n");
            output.push_str(&format!("path: {}\n", &rel));
            output.push_str(&format!(
                "name: {}\n",
                std::path::Path::new(&rel)
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
            ));
            output.push_str("</file_info>\n");
            output.push_str(&content);
            output.push_str("\n\n");
            files_included += 1;
        }
    }

    let bytes = output.len();
    Ok((
        output,
        AskStats {
            files: files_included,
            bytes,
        },
    ))
}

fn generate_repo_answer_via_gemini(contents: &[GeminiContent], model: &str) -> Result<String> {
    let api_key =
        std::env::var("GEMINI_API_KEY").map_err(|_| anyhow::anyhow!("GEMINI_API_KEY not set"))?;
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        model, api_key
    );

    let req = GeminiRequest {
        contents: contents.to_vec(),
        tools: None,
        tool_config: None,
    };
    let resp: GeminiResponse = ureq::post(&url)
        .set("Content-Type", "application/json")
        .send_json(serde_json::to_value(&req)?)
        .map_err(|e| anyhow::anyhow!("Gemini request failed: {}", e))?
        .into_json()
        .map_err(|e| anyhow::anyhow!("invalid Gemini JSON: {}", e))?;

    let text = resp
        .candidates
        .and_then(|mut v| v.pop())
        .and_then(|c| c.content)
        .and_then(|c| c.parts)
        .and_then(|mut parts| parts.pop())
        .and_then(|p| p.text)
        .unwrap_or_default()
        .trim()
        .to_string();
    if text.is_empty() {
        anyhow::bail!("empty response from model")
    } else {
        Ok(text)
    }
}

fn generate_repo_answer_stream_via_gemini(contents: &[GeminiContent], model: &str) -> Result<String> {
    use std::io::{BufRead, BufReader};
    let api_key =
        std::env::var("GEMINI_API_KEY").map_err(|_| anyhow::anyhow!("GEMINI_API_KEY not set"))?;
    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:streamGenerateContent?key={}&alt=sse",
        model, api_key
    );

    let req = GeminiRequest {
        contents: contents.to_vec(),
        tools: None,
        tool_config: None,
    };
    let resp = ureq::post(&url)
        .set("Content-Type", "application/json")
        .set("Accept", "text/event-stream")
        .send_json(serde_json::to_value(&req)?)
        .map_err(|e| anyhow::anyhow!("Gemini stream request failed: {}", e))?;

    let mut reader = BufReader::new(resp.into_reader());
    let inner = stream_box_start("Answer");
    let mut text_buf = String::new();
    let mut full_text = String::new();
    let mut sse_event = String::new();
    let mut line = String::new();
    let mut streamed_any = false;
    let mut last_usage: Option<serde_json::Value> = None;

    while reader.read_line(&mut line)? > 0 {
        let l = line.trim_end().to_string();
        line.clear();
        // SSE events end with a blank line
        if l.is_empty() {
            if sse_event.is_empty() {
                continue;
            }
            // Remove possible 'data: ' prefix occurrences (one per line)
            let data = sse_event
                .lines()
                .filter_map(|ln| ln.strip_prefix("data:").map(|rest| rest.trim()))
                .collect::<Vec<_>>()
                .join("");
            sse_event.clear();

            if data.is_empty() {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&data) {
                // Extract any text
                let mut appended = false;
                if let Some(cands) = v.get("candidates").and_then(|c| c.as_array()) {
                    for cand in cands {
                        if let Some(content) = cand.get("content") {
                            if let Some(parts) = content.get("parts").and_then(|p| p.as_array()) {
                                for part in parts {
                                    if let Some(t) = part.get("text").and_then(|t| t.as_str()) {
                                        text_buf.push_str(t);
                                        full_text.push_str(t);
                                        appended = true;
                                    }
                                }
                            }
                        }
                        if let Some(delta) = cand.get("delta") {
                            if let Some(t) = delta.get("text").and_then(|t| t.as_str()) {
                                text_buf.push_str(t);
                                full_text.push_str(t);
                                appended = true;
                            }
                        }
                    }
                }
                // Capture usage metadata if present
                if v.get("usageMetadata").is_some() {
                    last_usage = Some(v.clone());
                }

                if appended {
                    streamed_any = true;
                    while let Some(pos) = text_buf.find('\n') {
                        let line_text = text_buf[..pos].to_string();
                        stream_box_line(inner, &line_text);
                        text_buf.drain(..=pos);
                    }
                }
            }
            continue;
        }
        // accumulate event lines
        sse_event.push_str(&l);
        sse_event.push('\n');
    }
    if !text_buf.is_empty() {
        stream_box_line(inner, &text_buf);
    }
    stream_box_end(inner);
    if let Some(u) = last_usage {
        if let Some(total) = u
            .get("usageMetadata")
            .and_then(|m| m.get("totalTokenCount"))
            .and_then(|x| x.as_i64())
        {
            print_info(&format!("Total tokens used: {}", total));
        }
    }
    if !streamed_any {
        return Err(anyhow::anyhow!("no streamed content"));
    }
    Ok(full_text)
}

// -------- Leftover helpers --------

fn list_changed_files_vs_head(repo_dir: &Path) -> Result<Vec<String>> {
    let base = diff_base_ref(repo_dir);
    let out = run_in_repo(repo_dir, &["git", "diff", "--name-only", base])?;
    let files: Vec<String> = out
        .lines()
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect();
    Ok(files)
}

fn run_in_repo_strings(repo_dir: &Path, args: Vec<String>) -> Result<String> {
    let mut it = args.iter();
    let cmd = it.next().ok_or_else(|| anyhow::anyhow!("empty command"))?;
    let output = Command::new(OsStr::new(cmd))
        .args(&args[1..])
        .current_dir(repo_dir)
        .output()
        .with_context(|| format!("failed to run {:?}", args))?;
    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        Err(anyhow::anyhow!(
            "command {:?} failed: {}",
            args,
            stderr.trim()
        ))
    }
}

fn diff_context_for_files(
    repo_dir: &Path,
    files: &Vec<String>,
) -> Result<(String, String, String)> {
    let base = diff_base_ref(repo_dir);
    let mut name_status_args = vec![
        "git".to_string(),
        "diff".to_string(),
        "--name-status".to_string(),
        base.to_string(),
        "--".to_string(),
    ];
    let mut shortstat_args = vec![
        "git".to_string(),
        "diff".to_string(),
        "--shortstat".to_string(),
        base.to_string(),
        "--".to_string(),
    ];
    let mut diff_args = vec![
        "git".to_string(),
        "diff".to_string(),
        "-U3".to_string(),
        base.to_string(),
        "--".to_string(),
    ];
    for f in files {
        name_status_args.push(f.clone());
        shortstat_args.push(f.clone());
        diff_args.push(f.clone());
    }
    let name_status = run_in_repo_strings(repo_dir, name_status_args)?;
    let shortstat = run_in_repo_strings(repo_dir, shortstat_args)?;
    let diff_sample = truncate(&run_in_repo_strings(repo_dir, diff_args)?, 20_000);
    Ok((name_status, shortstat, diff_sample))
}

fn commit_files_with_ai(
    repo_dir: &Path,
    files: &Vec<String>,
    multi_progress: &MultiProgress,
) -> Result<()> {
    if files.is_empty() {
        return Ok(());
    }
    let pb = multi_progress.add(ProgressBar::new_spinner());
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg} [{elapsed_precise}]")
            .unwrap(),
    );
    pb.enable_steady_tick(std::time::Duration::from_millis(100));
    pb.set_message("Generating commit for leftovers...");

    let (name_status, shortstat, diff_sample) = diff_context_for_files(repo_dir, files)?;
    let prompt = build_commit_prompt_multiline(&name_status, &shortstat, &diff_sample);
    let msg = match generate_commit_message_via_gemini(&prompt) {
        Ok(m) => m,
        Err(_) => fallback_commit_message_multiline(&name_status, &shortstat),
    };
    pb.finish_with_message(format!(
        "{}",
        "Leftover commit proposal ready".to_string().green().bold()
    ));

    // Stage only these files and commit
    let mut add_args = vec![
        "git".to_string(),
        "add".to_string(),
        "-A".to_string(),
        "--".to_string(),
    ];
    for f in files {
        add_args.push(f.clone());
    }
    run_in_repo_strings(repo_dir, add_args)?;

    print_boxed("Leftover Commit", &msg);
    if let Some((subject, body)) = split_subject_body(&msg) {
        if body.trim().is_empty() {
            run_in_repo(repo_dir, &["git", "commit", "-m", subject.trim()])?;
        } else {
            run_in_repo(
                repo_dir,
                &["git", "commit", "-m", subject.trim(), "-m", body.trim()],
            )?;
        }
    } else {
        run_in_repo(repo_dir, &["git", "commit", "-m", msg.trim()])?;
    }
    Ok(())
}

// -------------------- Pretty printing helpers --------------------

fn print_title(title: &str) {
    let line = hr();
    println!("{}", line.clone().dark_grey());
    println!("{} {}", "»".cyan().bold(), title.bold());
    println!("{}", line.dark_grey());
}

fn print_success(msg: &str) {
    println!("{} {}", "✓".green().bold(), msg);
}
fn print_info(msg: &str) {
    println!("{} {}", "i".cyan().bold(), msg);
}
fn print_warn(msg: &str) {
    println!("{} {}", "!".yellow().bold(), msg);
}

fn hr() -> String {
    let width = terminal::size().map(|(w, _)| w as usize).unwrap_or(80);
    let w = width.clamp(40, 120);
    "─".repeat(w)
}

fn print_boxed(title: &str, content: &str) {
    let mut lines: Vec<String> = content.lines().map(|s| s.to_string()).collect();
    if lines.is_empty() {
        lines.push(String::new());
    }
    let max_line = lines.iter().map(|s| s.len()).max().unwrap_or(0);
    let title_str = format!(" {} ", title);
    let inner_width = max_line.max(title_str.len());
    let top = format!("┌{}┐", "─".repeat(inner_width));
    let mid_title = format!(
        "│{}{}│",
        title_str.as_str().bold(),
        " ".repeat(inner_width.saturating_sub(title_str.len()))
    );
    println!("{}", top);
    println!("{}", mid_title);
    println!("│{}│", " ".repeat(inner_width));
    for l in lines {
        let pad = inner_width.saturating_sub(l.len());
        println!("│{}{}│", l, " ".repeat(pad));
    }
    println!("└{}┘", "─".repeat(inner_width));
}

// Streaming box helpers
fn stream_box_start(title: &str) -> usize {
    let width = terminal::size()
        .map(|(w, _)| w as usize)
        .unwrap_or(80)
        .clamp(40, 120);
    let inner = width;
    println!("┌{}┐", "─".repeat(inner));
    let title_str = format!(" {} ", title).bold();
    let pad = inner.saturating_sub(strip_ansi_len(&title_str.to_string()));
    println!("│{}{}│", title_str, " ".repeat(pad));
    println!("│{}│", " ".repeat(inner));
    inner
}

fn stream_box_line(inner: usize, line: &str) {
    if line.len() <= inner {
        let pad = inner.saturating_sub(line.len());
        println!("│{}{}│", line, " ".repeat(pad));
        return;
    }
    // Soft-wrap long lines to the box width based on character count
    let mut start = 0usize;
    let bytes = line.as_bytes();
    while start < bytes.len() {
        // Find end index for this chunk without splitting UTF-8 characters
        let mut end = (start + inner).min(bytes.len());
        // Move end back to a char boundary
        while end > start && (bytes[end - 1] & 0b1100_0000) == 0b1000_0000 {
            end -= 1;
        }
        if end == start {
            end = (start + inner).min(bytes.len());
        }
        let chunk = &line[start..end];
        let pad = inner.saturating_sub(chunk.len());
        println!("│{}{}│", chunk, " ".repeat(pad));
        start = end;
    }
}

fn stream_box_end(inner: usize) {
    println!("└{}┘", "─".repeat(inner));
}

// Helper to approximate visible length ignoring simple ANSI sequences used by Stylize
fn strip_ansi_len(s: &str) -> usize {
    strip_ansi(s).len()
}
fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut bytes = s.as_bytes().iter().cloned();
    let mut in_esc = false;
    while let Some(b) = bytes.next() {
        if in_esc {
            if b == b'm' {
                in_esc = false;
            }
            continue;
        }
        if b == 0x1B {
            // ESC
            in_esc = true;
            continue;
        }
        out.push(b as char);
    }
    out
}

fn build_changes_summary_box(numstat: &str, shortstat: &str, max_rows: usize) -> String {
    let mut out = String::new();
    let mut rows = Vec::new();
    for (i, line) in numstat.lines().enumerate() {
        if i >= max_rows {
            break;
        }
        // format: added\tdeleted\tpath
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 3 {
            let added = parts[0];
            let deleted = parts[1];
            let path = parts[2];
            rows.push(format!("+{:>6}  -{:>6}  {}", added, deleted, path));
        }
    }
    out.push_str(shortstat.trim());
    out.push('\n');
    if !rows.is_empty() {
        out.push_str("\n");
        for r in rows {
            out.push_str(&r);
            out.push('\n');
        }
        if numstat.lines().count() > max_rows {
            out.push_str(&format!(
                "… and {} more files\n",
                numstat.lines().count() - max_rows
            ));
        }
    }
    out
}

// -------------------- First-run API key setup --------------------

fn ensure_gemini_api_key_interactive() -> Result<()> {
    if std::env::var("GEMINI_API_KEY").is_ok() {
        return Ok(());
    }

    print_warn(
        "GEMINI_API_KEY not set. AI commit messages require a Google Generative Language API key.",
    );
    println!("Get a key: {}", "https://ai.google.dev/".underlined());
    let input =
        rpassword::prompt_password("Enter GEMINI_API_KEY (hidden, or press Enter to skip): ")
            .map_err(|e| anyhow::anyhow!("failed to read input: {}", e))?;
    let key = input.trim().to_string();
    if key.is_empty() {
        print_warn("No key entered. AI commit requires GEMINI_API_KEY. Exiting.");
        return Err(anyhow::anyhow!("GEMINI_API_KEY not provided"));
    }

    // Set for current process
    std::env::set_var("GEMINI_API_KEY", &key);

    // Persist to shell RC
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    let shell = std::env::var("SHELL").unwrap_or_default();
    let mut rc_path = std::path::PathBuf::from(&home);
    if shell.contains("zsh") {
        rc_path.push(".zshrc");
    } else if shell.contains("bash") {
        rc_path.push(".bashrc");
    } else {
        // Default to zshrc if unknown
        rc_path.push(".zshrc");
    }

    let line = format!(
        "\n# repod: AI commit setup\nexport GEMINI_API_KEY=\"{}\"\n",
        key
    );
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&rc_path)
    {
        Ok(mut f) => {
            use std::io::Write as _;
            if let Err(e) = f.write_all(line.as_bytes()) {
                print_warn(&format!(
                    "Saved key for this session, but failed to update {}: {}",
                    rc_path.display(),
                    e
                ));
            } else {
                print_success(&format!("Saved GEMINI_API_KEY to {}", rc_path.display()));
            }
        }
        Err(e) => {
            print_warn(&format!(
                "Saved key for this session, but failed to open {}: {}",
                rc_path.display(),
                e
            ));
        }
    }

    Ok(())
}

// -------------------- Branch helpers --------------------

fn ensure_on_target_branch(
    repo_dir: &Path,
    branch_spec: Option<&str>,
    multi_progress: &MultiProgress,
) -> Result<String> {
    let current = get_current_branch(repo_dir)?;
    match branch_spec.map(|s| s.trim()) {
        None => Ok(current),
        Some(".") | Some("auto") => {
            // Generate a branch name
            let pb = multi_progress.add(ProgressBar::new_spinner());
            pb.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.green} {msg} [{elapsed_precise}]")
                    .unwrap(),
            );
            pb.enable_steady_tick(std::time::Duration::from_millis(100));
            pb.set_message("Generating branch name...");
            let suggested = generate_branch_name(repo_dir)
                .or_else(|_| heuristic_branch_name(repo_dir))
                .unwrap_or_else(|_| default_branch_name());
            pb.finish_with_message(format!("Proposed branch: {}", suggested));
            println!("");
            let choice = prompt_choice_keypress(
                "› Create branch? [y=accept, e=edit, n=stay]: ",
                &['y', 'e', 'n'],
            )?;
            match choice {
                'y' => {
                    switch_to_branch(repo_dir, &suggested, true)?;
                    Ok(suggested)
                }
                'e' => {
                    let edited = read_line_prompt(&format!("Enter branch name [{}]: ", suggested))?;
                    let name = if edited.trim().is_empty() {
                        suggested
                    } else {
                        sanitize_branch_name(&edited)
                    };
                    switch_to_branch(repo_dir, &name, true)?;
                    Ok(name)
                }
                _ => {
                    print_info("Staying on current branch.");
                    Ok(current)
                }
            }
        }
        Some(target) => {
            if target == current {
                return Ok(current);
            }
            // If target exists, switch; else create
            let exists = run_in_repo(repo_dir, &["git", "rev-parse", "--verify", target]).is_ok();
            switch_to_branch(repo_dir, target, !exists)?;
            Ok(target.to_string())
        }
    }
}

fn get_current_branch(repo_dir: &Path) -> Result<String> {
    let name = run_in_repo(repo_dir, &["git", "rev-parse", "--abbrev-ref", "HEAD"])?;
    Ok(name.trim().to_string())
}

fn switch_to_branch(repo_dir: &Path, name: &str, create: bool) -> Result<()> {
    // Stash if dirty
    let dirty = !run_in_repo(repo_dir, &["git", "status", "--porcelain"])?
        .trim()
        .is_empty();
    let mut stashed = false;
    if dirty {
        run_in_repo(repo_dir, &["git", "stash", "-u", "-q"])?;
        stashed = true;
    }
    let res = if create {
        run_in_repo(repo_dir, &["git", "checkout", "-b", name])
    } else {
        run_in_repo(repo_dir, &["git", "checkout", name])
    };
    if let Err(e) = res {
        return Err(e);
    }
    if stashed {
        // Try to restore
        let _ = run_in_repo(repo_dir, &["git", "stash", "pop", "-q"]);
    }
    print_success(&format!("On branch {}", name));
    Ok(())
}

fn try_push(repo_dir: &Path, branch: &str) -> Result<()> {
    print_info(&format!("Pushing branch '{}' to origin...", branch));
    let res = run_in_repo(repo_dir, &["git", "push", "-u", "origin", branch]);
    match res {
        Ok(out) => {
            println!("{}", out);
            print_success("Push complete.");
            Ok(())
        }
        Err(e) => {
            print_warn(&format!("Push failed: {}", e));
            Ok(())
        }
    }
}

fn generate_branch_name(repo_dir: &Path) -> Result<String> {
    // Use diff to propose a branch name via Gemini
    let diff_base = diff_base_ref(repo_dir);
    let name_status = run_in_repo(repo_dir, &["git", "diff", "--name-only", diff_base])?;
    let summary = run_in_repo(repo_dir, &["git", "diff", "--shortstat", diff_base])?;
    let prompt = format!(
        "Propose a short git branch name based on these changes.\n\
        Rules: lowercase, words separated by '-', prefix with a conventional type (feat|fix|chore|refactor|docs|test|perf), optional scope in words, max 48 chars total, no spaces, only [a-z0-9-].\n\
        Output ONLY the branch name.\n\
        Files:\n{}\n\
        Summary: {}",
        name_status.trim(), summary.trim()
    );
    let text = generate_commit_message_via_gemini(&prompt)?;
    Ok(sanitize_branch_name(&text))
}

fn heuristic_branch_name(repo_dir: &Path) -> Result<String> {
    let diff_base = diff_base_ref(repo_dir);
    let files = run_in_repo(repo_dir, &["git", "diff", "--name-only", diff_base])?;
    let first = files
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("changes");
    let scope = first.split('/').next().unwrap_or("changes");
    let date = chrono::Local::now().format("%Y%m%d");
    let base = format!("feat-{}-{}", scope, date);
    Ok(sanitize_branch_name(&base))
}

fn default_branch_name() -> String {
    let date = chrono::Local::now().format("%Y%m%d");
    format!("feat-changes-{}", date)
}

fn sanitize_branch_name(s: &str) -> String {
    let mut out = s.trim().to_lowercase();
    out = out
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '/' {
                c
            } else {
                '-'
            }
        })
        .collect();
    while out.contains("--") {
        out = out.replace("--", "-");
    }
    out.trim_matches('-').chars().take(48).collect()
}

fn is_text_file(path: &Path, repo_types: Option<&[RepoType]>) -> Result<bool> {
    // Always allow README files
    if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
        let name_lower = name.to_lowercase();
        if name_lower.contains("readme.") || name_lower == "readme" {
            return Ok(true);
        }
    }

    // If repo_types is specified, check if file matches any of the types
    if let Some(repo_types) = repo_types {
        let ext_lower = path
            .extension()
            .map(|ext| ext.to_string_lossy().to_lowercase());
        let file_lower = path
            .file_name()
            .and_then(|name| name.to_str())
            .map(|s| s.to_lowercase());

        return Ok(repo_types.iter().any(|repo_type| {
            let patterns = get_repo_type_extensions(repo_type);
            let ext_match = ext_lower
                .as_deref()
                .map_or(false, |ext| patterns.iter().any(|&p| p == ext));
            let file_match = file_lower
                .as_deref()
                .map_or(false, |name| patterns.iter().any(|&p| p == name));
            ext_match || file_match
        }));
    }

    // If no repo_types specified, use the original text file detection logic
    // Check if it's a known text extension
    if let Some(ext) = path.extension() {
        let ext_str = ext.to_string_lossy().to_lowercase();
        if TEXT_EXTENSIONS.contains(&ext_str.as_str()) {
            return Ok(true);
        }
    }

    // Use file signature detection
    if let Some(kind) = infer::get_from_path(path)? {
        let mime = kind.mime_type();
        // Known text MIME types
        if mime.starts_with("text/") || mime == "application/json" || mime == "application/xml" {
            return Ok(true);
        }
        // Known binary MIME types
        if mime.starts_with("image/")
            || mime.starts_with("audio/")
            || mime.starts_with("video/")
            || mime.starts_with("application/octet-stream")
            || mime.starts_with("application/x-executable")
        {
            return Ok(false);
        }
    }

    // If we can't determine by MIME type, analyze content
    let mut file = File::open(path)?;
    let mut buffer = vec![0; BINARY_CHECK_SIZE];
    let n = file.read(&mut buffer)?;
    if n == 0 {
        return Ok(true); // Empty files are considered text
    }

    // Count control characters and high ASCII
    let non_text = buffer[..n]
        .iter()
        .filter(|&&byte| {
            // Allow common control chars: tab, newline, carriage return
            byte != b'\t' &&
                byte != b'\n' &&
                byte != b'\r' &&
                // Consider control characters and high ASCII as non-text
                (byte < 32 || byte > 126)
        })
        .count();

    // Calculate ratio of non-text bytes
    let ratio = (non_text as f32) / (n as f32);
    Ok(ratio <= TEXT_THRESHOLD)
}

fn should_process_file(
    path: &Path,
    repo_root: &Path,
    repo_types: Option<&[RepoType]>,
    only_set: Option<&GlobSet>,
    exclude_set: Option<&GlobSet>,
) -> bool {
    let rel = normalize_rel_path(path, repo_root);
    // If only globs exist, require a match on the repo-relative path
    if let Some(set) = only_set {
        if !set.is_match(&rel) {
            return false;
        }
    }

    if let Some(set) = exclude_set {
        if set.is_match(&rel) {
            return false;
        }
    }

    // Then continue with regular filtering by repo_types/textness
    match is_text_file(path, repo_types) {
        Ok(is_text) => is_text,
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bin_pattern_does_not_match_ingest_bin_paths() {
        let custom = Vec::new();
        let set = build_exclude_globset(EXCLUDED_PATTERNS, &custom).expect("exclude set");
        assert!(set.is_match("bin/foo.rs"));
        assert!(!set.is_match("ingest_bin/src/lib.rs"));
        assert!(!set.is_match("tmp_bind.rs"));
        assert!(!set.is_match("src/main.rs"));
    }
}
fn extract_repo_name(url: &str) -> String {
    url.split('/')
        .last()
        .unwrap_or("repo")
        .trim_end_matches(".git")
        .to_string()
}

fn is_binary_file(path: &Path) -> Result<bool> {
    // First check if we can detect the file type. Prefer an explicit allow/deny
    // list rather than assuming every non-`text/` MIME is binary because many
    // textual assets are tagged as `application/*` (Package manifests, JSON, etc.).
    if let Some(kind) = infer::get_from_path(path)? {
        let mime = kind.mime_type();
        let is_text_mime = mime.starts_with("text/")
            || matches!(
                mime,
                "application/json"
                    | "application/ld+json"
                    | "application/xml"
                    | "application/javascript"
                    | "application/x-javascript"
                    | "application/sql"
                    | "application/yaml"
                    | "application/toml"
                    | "application/graphql"
                    | "application/x-sh"
            );
        if is_text_mime {
            return Ok(false);
        }

        let is_known_binary = mime.starts_with("image/")
            || mime.starts_with("audio/")
            || mime.starts_with("video/")
            || mime == "application/octet-stream"
            || mime == "application/pdf"
            || mime == "application/zip"
            || mime == "application/x-executable";
        if is_known_binary {
            return Ok(true);
        }
    }

    // If we can't detect the type, try to read the first few bytes
    // to check for null bytes (common in binary files)
    let mut file = File::open(path)?;
    let mut buffer = [0; 512];
    let n = file.read(&mut buffer)?;

    // Check for null bytes in the first chunk of the file
    Ok(buffer[..n].contains(&0))
}

fn print_stats(stats: &ProcessingStats) {
    println!("\nProcessing Statistics:");
    println!("Total repositories processed: {}", stats.repo_count);
    println!("Total files processed: {}", stats.total_files);
    println!("Total binary files skipped: {}", stats.binary_files_skipped);
    println!("Total tokens: {}", stats.total_tokens);
    println!("Repository clone time: {:.2} seconds", stats.clone_time);
    println!(
        "Content processing time: {:.2} seconds",
        stats.processing_time
    );
    println!(
        "Total time: {:.2} seconds",
        stats.clone_time + stats.processing_time
    );
    println!(
        "Average tokens per file: {:.2}",
        (stats.total_tokens as f64) / (stats.total_files as f64)
    );
    println!(
        "Processing speed: {:.2} files/second",
        (stats.total_files as f64) / stats.processing_time
    );
}
