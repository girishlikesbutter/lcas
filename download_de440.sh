#!/bin/bash

# Download script for de440.bsp ephemeris file
# This file is required for accurate planetary positions

SPICE_DIR="data/spice_kernels/generic/spk"
FILENAME="de440.bsp"
URL="https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp"

echo "Downloading DE440 ephemeris file..."

# Create directory if it doesn't exist
mkdir -p "$SPICE_DIR"

# Download the file
if command -v curl &> /dev/null; then
    curl -L -o "$SPICE_DIR/$FILENAME" "$URL"
elif command -v wget &> /dev/null; then
    wget -O "$SPICE_DIR/$FILENAME" "$URL"
else
    echo "Error: Neither curl nor wget is installed. Please install one of them."
    exit 1
fi

# Check if download was successful
if [ -f "$SPICE_DIR/$FILENAME" ]; then
    echo "Successfully downloaded $FILENAME to $SPICE_DIR/"
    ls -lh "$SPICE_DIR/$FILENAME"
else
    echo "Error: Failed to download $FILENAME"
    exit 1
fi