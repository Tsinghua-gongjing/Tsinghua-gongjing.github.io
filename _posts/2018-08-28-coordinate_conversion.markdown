---
layout: post
category: "genomics"
title:  "Convert coordinates"
tags: [genomics, coordinate, format]
---

### [liftOver](https://genome.sph.umich.edu/wiki/LiftOver) command usage

```bash
[zhangqf7@loginview02 danRer7]$ liftOver
liftOver - Move annotations from one assembly to another
usage:
   liftOver oldFile map.chain newFile unMapped
oldFile and newFile are in bed format by default, but can be in GFF and
maybe eventually others with the appropriate flags below.
The map.chain file has the old genome as the target and the new genome
as the query.

***********************************************************************
WARNING: liftOver was only designed to work between different
         assemblies of the same organism. It may not do what you want
         if you are lifting between different organisms. If there has
         been a rearrangement in one of the species, the size of the
         region being mapped may change dramatically after mapping.
***********************************************************************

options:
   -minMatch=0.N Minimum ratio of bases that must remap. Default 0.95
   -gff  File is in gff/gtf format.  Note that the gff lines are converted
         separately.  It would be good to have a separate check after this
         that the lines that make up a gene model still make a plausible gene
         after liftOver
   -genePred - File is in genePred format
   -sample - File is in sample format
   -bedPlus=N - File is bed N+ format
   -positions - File is in browser "position" format
   -hasBin - File has bin value (used only with -bedPlus)
   -tab - Separate by tabs rather than space (used only with -bedPlus)
   -pslT - File is in psl format, map target side only
   -ends=N - Lift the first and last N bases of each record and combine the
             result. This is useful for lifting large regions like BAC end pairs.
   -minBlocks=0.N Minimum ratio of alignment blocks or exons that must map
                  (default 1.00)
   -fudgeThick    (bed 12 or 12+ only) If thickStart/thickEnd is not mapped,
                  use the closest mapped base.  Recommended if using
                  -minBlocks.
   -multiple               Allow multiple output regions
   -minChainT, -minChainQ  Minimum chain size in target/query, when mapping
                           to multiple output regions (default 0, 0)
   -minSizeT               deprecated synonym for -minChainT (ENCODE compat.)
   -minSizeQ               Min matching region size in query with -multiple.
   -chainTable             Used with -multiple, format is db.tablename,
                               to extend chains from net (preserves dups)
   -errorHelp              Explain error messages
```

### [CrossMap](http://crossmap.sourceforge.net/) command usage

```bash
[zhangqf7@loginview02 danRer7]$ CrossMap.py
Program: CrossMap (v0.2.5)

Description:
  CrossMap is a program for convenient conversion of genome coordinates and
  genomeannotation files between assemblies (eg. lift from human hg18 to hg19 or
  vice versa).It supports file in BAM, SAM, BED, Wiggle, BigWig, GFF, GTF and VCF
  format.

Usage: CrossMap.py <command> [options]

  bam   convert alignment file in BAM or SAM format.
  bed   convert genome cooridnate or annotation file in BED or BED-like format.
  bigwig        convert genome coordinate file in BigWig format.
  gff   convert genome cooridnate or annotation file in GFF or GTF format.
  vcf   convert genome coordinate file in VCF format.
  wig   convert genome coordinate file in Wiggle, or bedGraph format.
```

### Issues

#### 1. bigWig file reduced dramatically using CrossMap

CrossMap supports more file formats than liftOver, like wig/bigWig. However, when I use CrossMap to convert a bigWig file, the output file seems much smaller.

```bash
# download phastCons score of zebrafish (only z7 provide)
# link: //hgdownload.cse.ucsc.edu/goldenPath/danRer7/phastCons8way/
# currently there is no z10 version: https://www.biostars.org/p/282330/

# convert from z7 to z10
# output: fish.phastCons8way.bw.bgr, fish.phastCons8way.bw.bw, fish.phastCons8way.bw.sorted.bgr 
[zhangqf7@bnode02 conservation]$ CrossMap.py bigwig danRer7ToDanRer10.over.chain.gz danRer7/fish.phastCons8way.bw danRer10/fish.phastCons8way.bw
@ 2018-08-27 20:39:45: Read chain_file:  danRer7ToDanRer10.over.chain.gz
@ 2018-08-27 20:39:49: Liftover bigwig file: danRer7/fish.phastCons8way.bw ==> danRer10/fish.phastCons8way.bw.bgr
@ 2018-08-27 20:51:54: Merging overlapped entries in bedGraph file ...
@ 2018-08-27 20:51:54: Sorting bedGraph file:danRer10/fish.phastCons8way.bw.bgr
@ 2018-08-27 20:53:42: Convert wiggle to bigwig ...

# the converted .bw file is much smaller
[zhangqf7@loginview02 conservation]$ ll danRer7/fish.phastCons8way.bw
-rw-rw----+ 1 zhangqf7 zhangqf 757M Jan 20  2011 danRer7/fish.phastCons8way.bw
[zhangqf7@loginview02 conservation]$ ll danRer10/fish.phastCons8way.bw.bw
-rw-rw----+ 1 zhangqf7 zhangqf 22M Aug 27 20:53 danRer10/fish.phastCons8way.bw.bw

# convert z7 .bw to .bedgraph directly to check the line number
[zhangqf7@bnode02 danRer7]$ bigWigToBedGraph fish.phastCons8way.bw fish.phastCons8way.bgr
[zhangqf7@bnode02 danRer7]$ ll
total 8.1G
-rw-rw----+ 1 zhangqf7 zhangqf 5.5G Aug 27 21:59 fish.phastCons8way.bgr
-rw-rw----+ 1 zhangqf7 zhangqf 757M Jan 20  2011 fish.phastCons8way.bw
[zhangqf7@loginview02 conservation]$ wl danRer7/fish.phastCons8way.bgr
201830335 danRer7/fish.phastCons8way.bgr

# the convertd .bedgraph file line number is much smaller
[zhangqf7@loginview02 conservation]$ wl danRer10/fish.phastCons8way.bw.sorted.bgr
3213494 danRer10/fish.phastCons8way.bw.sorted.bgr
```  