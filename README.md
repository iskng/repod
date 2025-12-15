# repod

A CLI tool that recursively processes repositories, generates directory structures, and outputs repository contents in a format optimized for analysis.

Note: Default behavior changed recently — single-target runs copy to clipboard by default; multi-target runs write files. See “Default Behavior”.

## Features

- Clone and process Git repositories (HTTPS or SSH)
- Process local directories
- Generate directory tree structures 
- Filter files by language/repository type
- Exclude directories or file patterns
- Copy output to clipboard or save to file
- Detect and skip binary files
- Process large repositories efficiently with parallel processing
- Respects `.gitignore` files at all directory levels
- Automatically excludes hidden files and directories (starting with `.`)

## Installation

```bash
# Build from source
git clone https://github.com/yourusername/repod.git
cd repod
cargo build --release

# After building with cargo build --release
# Move to /usr/local/bin (requires sudo on most systems)
sudo cp target/release/repod /usr/local/bin/

# OR to user's bin directory (no sudo required)
mkdir -p ~/.local/bin
cp target/release/repod ~/.local/bin/

# Make sure ~/.local/bin is in your PATH
# Add this line to your .bashrc, .zshrc, or other shell config if needed:
# export PATH="$HOME/.local/bin:$PATH"

# The binary will be in target/release/repod
```

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines. For security concerns, see SECURITY.md.

## Usage

```bash
# Basic usage (process current directory; copies to clipboard by default)
repod

# Clone and process a GitHub repository
repod https://github.com/username/repo

# Clone with SSH
repod git@github.com:username/repo.git

# Write output to file (instead of copying)
repod --write

# Write to a custom output directory (also implies write mode)
repod -o custom_output

# Specify repository types to filter
repod -t rust,python,javascript

# Exclude specific directories or patterns
repod -e node_modules,target,build

# Only include files matching specific patterns
repod --only "*.mdx,*.tsx"

# Explicitly copy to clipboard (normally the default for single-target runs)
repod --copy

# Clone to a specific location
repod https://github.com/username/repo --at /path/to/clone

# Open in Cursor IDE after cloning
repod git@github.com:username/repo.git --open-cursor

# Stage and commit changes with AI (single commit)
# Note: --commit only works on the current directory
repod --commit

# Ask a question about the current repository (uses Gemini 2.5 Pro)
repod --ask "What are the main components and how do they interact?"

# Ask about a remote repository (HTTPS or SSH URL)
repod https://github.com/username/repo --ask "Summarize architecture and key modules"

# Copy the model's answer from --ask to the clipboard
repod --ask "Explain the build pipeline" --copy

# Commit to a specific branch (creates/switches if needed)
repod --commit=feature/my-branch

# Let repod propose a branch name from changes (confirm or edit before creating)
repod --commit=auto

# Propose and apply multiple commits (current directory only)
repod --multi-commit

```

## Authentication

For private repositories:

```bash
# Using GitHub token
repod https://github.com/username/private-repo --github-token YOUR_TOKEN

# Or set the token as an environment variable
export GITHUB_TOKEN=your_token
repod https://github.com/username/private-repo

# Using SSH with a custom key
repod git@github.com:username/private-repo.git --ssh-key ~/.ssh/custom_key
```

## Options

```
Options:
  -o, --output-dir <OUTPUT_DIR>  Output directory path [default: output] (implies write mode if set)
  -t, --repo-types <REPO_TYPES>  Repository types to filter files (e.g., rs, py, js, ts)
  -p, --github-token <GITHUB_TOKEN>  GitHub personal access token for private repositories
  -e, --exclude <EXCLUDE>        Additional folder or path patterns to exclude from processing
      --only <ONLY>              Only include files matching these patterns (e.g., *.mdx, *.tsx)
      --only-dir <ONLY_DIR>      Only include files under these directories (relative to repo root). Multiple via comma.
                                 Examples: --only-dir src,docs (includes everything under src/ and docs/)
                                 Notes: equivalent to adding `<dir>/**` to --only; combine with --only to refine types.
      --ssh-key <SSH_KEY>        SSH key path (defaults to ~/.ssh/id_rsa)
      --ssh-passphrase <SSH_PASSPHRASE>  SSH key passphrase (if not provided, will prompt if needed)
      --open-cursor              Open in Cursor after cloning
      --at <AT>                  Specific path to clone the repository to
      --copy                     Copy output to clipboard (explicit)
      --write                    Write output to file (overrides default copy behavior)
      --commit                   Single AI-generated commit (current dir only)
      --multi-commit            AI-proposed multi-commit plan (current dir only)
      --branch <BRANCH>         Target branch: name or 'auto' to propose one
      --push                     After committing, push the current branch to 'origin' (sets upstream if needed)
      --ask <QUESTION>           Ask a question about the current repository (uses Gemini 2.5 Pro)
  -h, --help                     Print help
  -V, --version                  Print version
```

## Default Behavior

- Single target (no CSV; one repo or current dir): copies output to clipboard by default.
- Multiple targets (CSV or multiple URLs): writes output files by default to avoid clipboard races.
- If `-o/--output-dir` is provided, the tool writes to files unless `--copy` is explicitly passed.
- Use `--write` to force writing; use `--copy` to force copying.

Pattern semantics: `--only` uses globset-style globs with real `**` recursion. Examples: `**/*.rs`, `src/**`, `docs/**/*.md`. Bare patterns like `*.rs` are treated as `**/*.rs` (match in any directory). 

## Output Format

The output contains:
- A directory structure section with a tree view of the repository
- File contents with path information
- Files are processed in chunks to handle large repositories efficiently

## Examples

### Basic Repository Processing

```bash
repod https://github.com/username/repo -o analysis
```

### Filter by Language and Exclude Directories

```bash
repod -t rust,go -e tests,examples,target
```

### Process Current Directory (Default Copy)

```bash
repod
```

### Process Current Directory and Write to File

```bash
repod --write
```

### Process Only Specific File Types

```bash
# Only include MDX and TSX files
repod --only "*.mdx,*.tsx"

# Only include files in specific directories matching a pattern
repod --only "src/**/*.rs,tests/**/*.rs"

### Include Only Specific Directories

```bash
# Include only the src/ and docs/ trees (all files under them)
repod --only-dir src,docs

# Combine with --only for file-type refinement
repod --only-dir src --only "*.rs,*.toml"
```

Notes on pattern semantics:
- `--only` uses globset globs (gitignore-like). `**` is recursive; `*` matches within a segment. Bare patterns like `*.rs` match anywhere (internally expanded to `**/*.rs`).
- `--only-dir` works like adding `<dir>/**` to `--only`. For nested paths, pass e.g. `--only-dir src/lib`.
```

## CSV Input

Provide a CSV file with repository URLs in the first column to process multiple repositories in parallel:

```bash
repod repos.csv --write
```

Notes:
- With CSV or multiple URLs, default is to write files to avoid clipboard overwrites.
- You can still force clipboard behavior with `--copy` (last finisher wins in the clipboard).

## AI Commit Messages

When `--commit` is provided, the tool proposes a Conventional Commit message with a subject and a short body based on your current diff (against `HEAD`). It uses Google’s Gemini model `models/gemini-2.5-flash` via the Generative Language API. You’ll be shown the message in a clean, boxed view and asked to confirm with a single keypress (press `y` to commit, `n`/Esc to cancel — no Enter needed).

First run: If `GEMINI_API_KEY` is not set, repod prompts you to paste it (input is hidden). If provided, it saves the key to your shell config (`~/.zshrc` for zsh or `~/.bashrc` for bash) and uses it immediately for the current session. If you skip providing a key, the command exits — there is no local fallback when the API key is missing.

Branch selection:
- Without `--branch`, commits use the currently checked-out branch.
- With `--branch <name>`, repod creates/switches to `<name>` if needed before committing.
- With `--branch auto`, repod proposes a branch name from your changes; you can accept or edit the name before creating.

UI details:
- The banner shows the current mode and branch (e.g., `AI Commit (Single) — branch: feature/foo`).
- Proposed commit messages are shown in a boxed view; confirm with a single keypress (`y`/`n`, no Enter).
- For `--multi-commit`, each proposed commit is confirmed one-by-one with a single keypress.
- A "Changes" summary box is shown with added/deleted lines per file (from `git diff --numstat`), plus the `--shortstat` summary.

## Ask About Repository (`--ask`)

Provides a repository-wide context (directory tree + selected file contents) and sends your question to `models/gemini-3-pro-preview`. The tool uses your `.gitignore`, built-in exclusions, and `--only`/`--exclude` filters. On very large repos, the context may be truncated to fit model limits. Answers stream live in a boxed view; usage stats are printed (files included, context bytes, prompt tokens, prep time). After each answer you can type a follow-up; the repository dump is reused and the conversation history is sent, so context carries across turns. Press Enter or `q`/`:q` to exit.

Limits:
- For safety, the conversation (repo context + history) is capped at ~1,000,000 tokens. If the limit is exceeded, the request is not sent. Narrow scope with `--only`/`--exclude`.

Example:

```bash
repod --ask "List the main services and where their HTTP routes are defined."
```

Environment variable:

```bash
export GEMINI_API_KEY=your_google_api_key
```

Clipboard:
- By default `--ask` does not copy anything. Add `--copy` to copy the full model answer to the clipboard after printing.

Remote repositories with `--ask`:
- You can pass a single HTTPS/SSH repo URL with `--ask`. The repo is cloned into a temporary directory and analyzed. CSV inputs are not supported with `--ask`.

### Multi-Commit Planning (`--multi-commit`)

The tool analyzes all changes and proposes a set of smaller, logical commits. It shows a styled summary of each proposed commit (title, optional body, and files included), then asks you to confirm each commit one-by-one with a single keypress (`y` to apply, `n` to skip). Any remaining files can optionally be committed at the end.

Notes:
- Planning uses file-level grouping (not hunk-level).
- You can review the plan before execution and cancel if it doesn’t look right.

## Exclusions

The tool automatically excludes many common directories and lock files (e.g., `.git/`, `node_modules/`, `target/`, build caches, and lockfiles like `Cargo.lock`, `yarn.lock`, `package-lock.json`). Hidden files and directories (names starting with `.`) are skipped. You can add more exclusions with `-e/--exclude`.

Cursor mode note: when `--open-cursor` is used and writing is enabled, the output file is written into the repo root as `screenpipe_<timestamp>.txt` and Cursor is launched pointing at the repo.
