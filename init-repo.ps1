# --- CONFIGURATION ---
$repoName = "scraple-solver"
$description = "Python script for solving Scraple daily puzzles"
$isPrivate = $false  # Set to $true to make the repo private

# --- STEP 1: Initialise Git ---
git init

# Exclude this script from git staging
$scriptName = $MyInvocation.MyCommand.Name
git add --all
git reset HEAD $scriptName  # Unstage this script

git commit -m "Initial commit"

# --- STEP 2: Create GitHub repo via CLI ---
$visibility = if ($isPrivate) { "--private" } else { "--public" }
gh repo create $repoName `
  --source=. `
  --remote=origin `
  $visibility `
  --description "$description" `
  --push

# --- STEP 3: Confirm success ---
Write-Host "`n🎉 Repo '$repoName' created and pushed to GitHub successfully!" -ForegroundColor Green
gh repo view --web