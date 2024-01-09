#!/bin/bash

# Function to check if the last command was successful
check_status() {
    if [ $? -eq 0 ]; then
        echo "Success"
    else
        echo "Error: Something went wrong. Exiting..."
        exit 1
    fi
}

# Prompt user for commit message or use a default one
read -p "Enter commit message (or press Enter for default): " commit_message
commit_message=${commit_message:-"Update"}

# Add changes to the staging area
git add .

# Check if there are changes to commit
if [[ $(git diff --cached --name-only) ]]; then
    # Commit changes with the provided/ default commit message
    git commit -m "$commit_message"
    check_status

    # Push changes to the main branch
    git push origin main
    check_status

    echo "Changes successfully committed and pushed."
else
    echo "No changes to commit. Exiting..."
fi
