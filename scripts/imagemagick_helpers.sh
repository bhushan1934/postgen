#!/bin/bash
# Test script to generate a gradient background
magick -size 1080x1080 gradient:'#667eea-#764ba2' test_gradient.jpg
echo "Gradient saved as test_gradient.jpg"
