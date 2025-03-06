#!/bin/bash

# Usage function
usage() {
    echo "Usage: $0 [OPTIONS] [PDB_FILE] [MODEL_TYPE] [OUTPUT_DIR]"
    echo
    echo "Options:"
    echo "  -i, --interactive     Run in interactive mode (default if no PDB file provided)"
    echo "  -c, --cpu             Run using CPU only (no GPU acceleration)"
    echo "  -h, --help            Show this help message"
    echo
    echo "Arguments:"
    echo "  PDB_FILE    Path to the PDB file to process"
    echo "  MODEL_TYPE  Type of model to use, must be one of:"
    echo "              - dna_vs_rna : Classify DNA vs RNA binding proteins"
    echo "              - dna_vs_non : Classify DNA vs non-nucleic acid binding proteins" 
    echo "              - rna_vs_non : Classify RNA vs non-nucleic acid binding proteins"
    echo "  OUTPUT_DIR  Optional directory to save output files"
    echo
    echo "If no PDB file is provided, the container will start in interactive mode."
    exit 1
}

# Default values
INTERACTIVE=false
USE_CPU=false
PDB_FILE=""
MODEL_TYPE=""
OUTPUT_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--interactive)
            INTERACTIVE=true
            shift
            ;;
        -c|--cpu)
            USE_CPU=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *.pdb)
            PDB_FILE="$1"
            shift
            ;;
        dna_vs_rna|dna_vs_non|rna_vs_non)
            MODEL_TYPE="$1"
            shift
            ;;
        *)
            # If PDB and MODEL_TYPE are set, and this is a path, treat as OUTPUT_DIR
            if [ -n "$PDB_FILE" ] && [ -n "$MODEL_TYPE" ] && [ -z "$OUTPUT_DIR" ]; then
                OUTPUT_DIR="$1"
            else
                echo "Unknown option or argument: $1"
                usage
            fi
            shift
            ;;
    esac
done

# Set the container name
CONTAINER_NAME="pnabind"

# GPU configuration
if [ "$USE_CPU" = true ]; then
    GPU_ARGS="--gpus 0"
    echo "Running in CPU-only mode (no GPU acceleration)"
else
    GPU_ARGS="--gpus all"
    echo "Running with GPU acceleration"
fi

# Determine run mode
if [ "$INTERACTIVE" = true ]; then
    # Run in interactive mode
    docker run $GPU_ARGS -it --rm \
        --name $CONTAINER_NAME \
        pnabind:latest
elif [ -n "$PDB_FILE" ]; then
    # Check if PDB file exists
    if [ ! -f "$PDB_FILE" ]; then
        echo "Error: PDB file '$PDB_FILE' not found."
        exit 1
    fi
    
    # Check if model type is provided
    if [ -z "$MODEL_TYPE" ]; then
        echo "Error: MODEL_TYPE must be specified when processing a PDB file."
        echo "Valid options: dna_vs_rna, dna_vs_non, rna_vs_non"
        exit 1
    fi
    
    # Get the absolute path of the PDB file
    PDB_FILE_ABS=$(realpath "$PDB_FILE")
    
    # Set up output directory
    OUTPUT_ARGS=""
    if [ -n "$OUTPUT_DIR" ]; then
        mkdir -p "$OUTPUT_DIR"
        OUTPUT_DIR_ABS=$(realpath "$OUTPUT_DIR")
        OUTPUT_ARGS="$OUTPUT_DIR_ABS"
    fi
    
    # Run with PDB file and model type
    docker run $GPU_ARGS --rm \
        --name $CONTAINER_NAME \
        -v "$PDB_FILE_ABS":"$PDB_FILE_ABS" \
        $([ -n "$OUTPUT_DIR" ] && echo "-v $OUTPUT_DIR_ABS:$OUTPUT_DIR_ABS") \
        pnabind:latest "$PDB_FILE_ABS" "$MODEL_TYPE" "$OUTPUT_ARGS"
else
    # Default to interactive mode if no arguments
    docker run $GPU_ARGS -it --rm \
        --name $CONTAINER_NAME \
        pnabind:latest
fi
