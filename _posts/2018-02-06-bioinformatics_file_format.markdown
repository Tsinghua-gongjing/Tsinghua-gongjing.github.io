---
layout: post
category: "genomics"
title:  "Common file format in bioinformatics"
tags: [genomics, bioinformatics, format]
---

### bed file

Full description can be accessed at [UCSC bed](http://genome.ucsc.edu/FAQ/FAQformat#format1), here are example from [bedtools introduction](https://bedtools.readthedocs.io/en/latest/content/general-usage.html#bed-format) :

columns: 12 (some are optional correspond to different style)

1. **chrom** - The name of the chromosome on which the genome feature exists.
2. **start** - The 0-based starting position of the feature in the chromosome.
3. **end** - The one-based ending position of the feature in the chromosome.
4. **name** - Defines the name of the BED feature.
5. **score** - The UCSC definition requires that a BED score range from 0 to 1000, inclusive.
6. **strand** - Defines the strand - either ‘+’ or ‘-‘.
7. **thickStart** - The starting position at which the feature is drawn thickly.
8. **thickEnd** - The ending position at which the feature is drawn thickly.
9. **itemRgb** - An RGB value of the form R,G,B (e.g. 255,0,0).
10. **blockCount** - The number of blocks (exons) in the BED line.
11. **blockSizes** - A comma-separated list of the block sizes.
12. **blockStarts** - A comma-separated list of block starts.

![bed](/assets/bed_file_format_example.jpeg)