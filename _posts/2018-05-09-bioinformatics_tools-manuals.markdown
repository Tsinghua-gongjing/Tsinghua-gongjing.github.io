---
layout: post
category: "genomics"
title:  "Manuals of bioinformatics tools"
tags: [genomics]
---

## list of tools

|Tool|Manual|Description|
|---|---|---|
|fastqc|[pdf](), [online](https://www.bioinformatics.babraham.ac.uk/projects/fastqc/)|provide a simple way to do some quality control checks on raw sequence data coming from high throughput sequencing pipelines.|
|cutadapt|[pdf](), [online](https://cutadapt.readthedocs.io/en/stable/guide.html)|Cutadapt finds and removes adapter sequences, primers, poly-A tails and other types of unwanted sequence from your high-throughput sequencing reads.|
|Trimmomatic|[pdf](https://github.com/Tsinghua-gongjing/blog_codes/blob/master/files/manuals/TrimmomaticManual_V0.32.pdf), [online](http://www.usadellab.org/cms/uploads/supplementary/Trimmomatic/TrimmomaticManual_V0.32.pdf)|Trimmomatic is a fast, multithreaded command line tool that can be used to trim and crop Illumina(FASTQ) data as well as to remove adapters.|
|STAR|[pdf](https://github.com/Tsinghua-gongjing/blog_codes/blob/master/files/manuals/STARmanual.pdf), [online](https://github.com/alexdobin/STAR)|To align our large (>80 billon reads) ENCODE Transcriptome RNA-seq dataset, we developed the Spliced Transcripts Alignment to a Reference (STAR) software based on a previously undescribed RNA-seq alignment algorithm that uses sequential maximum mappable seed search in uncompressed suffix arrays followed by seed clustering and stitching procedure.|
|Bowtie2|[pdf](), [online](http://bowtie-bio.sourceforge.net/bowtie2/manual.shtml)|Bowtie 2 is an ultrafast and memory-efficient tool for aligning sequencing reads to long reference sequences. It is particularly good at aligning reads of about 50 up to 100s or 1,000s of characters to relatively long (e.g. mammalian) genomes.|
|SAM file format|[pdf](https://github.com/Tsinghua-gongjing/blog_codes/blob/master/files/manuals/SAMv1_file_format_manual.pdf), [online](https://samtools.github.io/hts-specs/SAMv1.pdf)|SAM stands for Sequence Alignment/Map format.  It is a TAB-delimited text format consisting of a header section, which is optional, and an alignment section.  If present, the header must be prior to the alignments.Header lines start with ‘@’, while alignment lines do not.  Each alignment line has 11 mandatory fields for essential alignment information such as mapping position, and variable number of optional fields for flexible or aligner specific information.|
|Bedtools|[pdf](), [online](https://bedtools.readthedocs.io/en/latest/)|Collectively, the bedtools utilities are a swiss-army knife of tools for a wide-range of genomics analysis tasks. The most widely-used tools enable genome arithmetic: that is, set theory on the genome. For example, bedtools allows one to intersect, merge, count, complement, and shuffle genomic intervals from multiple files in widely-used genomic file formats such as BAM, BED, GFF/GTF, VCF. While each individual tool is designed to do a relatively simple task (e.g., intersect two interval files), quite sophisticated analyses can be conducted by combining multiple bedtools operations on the UNIX command line.|


