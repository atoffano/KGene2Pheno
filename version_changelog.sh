#!/bin/bash

# Take in the release version as a command line argument
# Get the version release number from user input
read -p "Enter the WS version release number : " release
release=$(echo $release | sed -E 's/^ws|^WS//i')
prev_release=`expr $release - 1`

# Build the URLs for the two changelog files we want to compare
changelog_url="https://downloads.wormbase.org/releases/WS${release}/letter.WS${release}"
prev_changelog_url="https://downloads.wormbase.org/releases/WS${prev_release}/letter.WS${prev_release}"

# Download the two changelog files
curl -o current_changelog.txt $changelog_url
curl -o prev_changelog.txt $prev_changelog_url

# Print the summary of changes
echo ""
echo "Summary of changes between WS${prev_release} and WS${release}:"
echo "=============================================================="
echo 
echo "-========================= WS${prev_release} ============================   -========================= WS${release} ============================"
echo "" 

# Use the `diff` command to compare the two files and format the output
sdiff --text  prev_changelog.txt current_changelog.txt

# Clean up the downloaded files
rm current_changelog.txt
rm prev_changelog.txt
