all: 
	
	hg38_100kb.bed conserved.bed.gz 

HapmapII_GRCh37_RecombinationHotspots.tar.gz hapmap/genetic_map_GRCh37_chr10.txt sex-averaged.bed rec_100kb.bed hapmap_genetic_map.bed 

rec_100kb_chr10.bed conserved_chr10.bed neutral_chr10.bed conserved_by_chrom/conserved_chr10.bed.gz 

clean-all: clean-decode clean-hapmap clean-simplified clean-chr10 clean-simplified
 
## === Human Annotation Data

hg38_seqlens.tsv:
	curl https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.chrom.sizes > $@

phastConsElements100way.bed.gz:
		curl http://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/phastConsElements100way.txt.gz | zcat | bioawk -t '{print $$2,$$3,$$4}' | gzip > $@

hg38.ncbiRefSeq_exons.bed.gz:
	curl https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/genes/hg38.ncbiRefSeq.gtf.gz | bioawk -cgff '{ if ($$feature == "exon") { print $$seqname,$$start,$$end,$$feature }}' | sort --version-sort -k1,1 -k2,2n | gzip > $@

## === Genomic Windows

hg38_100kb.bed: hg38_seqlens.tsv
	bedtools makewindows -g $<  -w 100000 > $@


## === DeCode Recombination Map
sex-averaged.rmap: 
	wget https://www.decode.com/additional/sex-averaged.rmap

# bioawk is picky about the numeric column being an integer, so we scale up
sex-averaged.bed: sex-averaged.rmap
	sed 1d $< | bioawk -t '{ print $$1,$$2,$$2+10000,".",int($$4*100000) }' | sort -k1,1 -k2,2n > $@

# scales column back down to float
rec_100kb.bed: sex-averaged.bed hg38_100kb.bed
	bedtools map -a hg38_100kb.bed -b sex-averaged.bed -c 5 -o mean | bioawk -t '{ print $$1, $$2, $$3, $$4/100000 }' > $@

clean-decode: 
	rm -f sex-averaged.map sex-averaged.bed rec_100kb.bed 

## === HapMap Recombination Map 
HapmapII_GRCh37_RecombinationHotspots.tar.gz:
	wget ftp://ftp-trace.ncbi.nih.gov/1000genomes/ftp/technical/working/20110106_recombination_hotspots/HapmapII_GRCh37_RecombinationHotspots.tar.gz

# this untars all chroms
hapmap/genetic_map_GRCh37_chr10.txt: HapmapII_GRCh37_RecombinationHotspots.tar.gz
	mkdir -p hapmap
	tar -xvzf $< -C hapmap

# this is just a placeholder file; this works on the whole directory
hapmap_genetic_map.bed: hapmap/genetic_map_GRCh37_chr10.txt
	find hapmap/ -name "genetic_map_GRCh37_chr*txt" | grep -P "chr[0-9]+.txt" | sort --version-sort | xargs -n 1 cat | grep -v Chromosome | bioawk -t '{ print $$1,$$2,$$3 }' > $@

clean-hapmap:
	rm -f HapmapII_GRCh37_RecombinationHotspots.tar.gz hapmap/* hapmap_genetic_map.bed

## === Simplified Tracks for Sims
# merged exons/phastcons "conservation" track -- mostly a place holder for simulations
conserved.bed.gz:  hg38.ncbiRefSeq_exons.bed.gz phastConsElements100way.bed.gz
	(zcat hg38.ncbiRefSeq_exons.bed.gz; (zcat phastConsElements100way.bed.gz | \
	 bioawk -t '{ print $$1,$$2,$$3,"conserved" }')) | grep -w '^#\|chr[1-9]\|chr[1-2][0-9]\|chr[XY]' | \
	 sort  --version-sort -k 1,1 -k2,2n | \
	 bedtools merge  -c 4 -o distinct | gzip > conserved.bed.gz

conserved_slop.bed.gz: conserved.bed.gz hg38_seqlens.tsv
	zgrep -P "^chr[0-9]+\t" $< | sort --version-sort -k 1,1 -k2,2n | bedtools slop -g hg38_seqlens.tsv -b 1000 | bedtools merge -d 2  | gzip > $@

clean-simplified: 
	rm -f conserved.bed.gz conserved_slop.bed.gz

## === Rules for the chr10 SLiM Simulations

rec_100kb_chr10.bed: rec_100kb.bed
	grep "^chr10" $< | bioawk '{ if ($$3 <= 130000000) { print $$0} }' > $@

conserved_chr10.bed: conserved.bed.gz
	zgrep "chr10" $< | bioawk -t '{ if ($$3 < 130000000) { print $$1,$$2,$$3,"conserved" } }' | sort  --version-sort -k 1,1 -k2,2n |\
		bedtools slop -g hg38_seqlens.tsv -b 1000 | sort  --version-sort -k 1,1 -k2,2n | bedtools merge -d 2 > $@

# simpler chr10 for sims -- approximation is fine
chr10_seqlens.tsv:
	echo "chr10	130000000" > $@

# putatively neutral regions -- not conserved tracks
neutral_chr10.bed: conserved_chr10.bed chr10_seqlens.tsv
	bedtools complement -i $< -g chr10_seqlens.tsv | grep "^chr10" > $@

# sepearate out conserved tracks into separate chromosome files
conserved_by_chrom/conserved_chr10.bed.gz: conserved_slop.bed.gz
	mkdir -p conserved_by_chrom/
	for i in $$(seq 1 22); do zgrep -P "chr$$i\t" $< > conserved_by_chrom/chr$$i.bed; done

clean-chr10: 
	rm -rf rec_100kb_chr10.bed conserved_slop.bed chr10_seqlens.tsv neutral_chr10.bed conserved_by_chrom/* 



# B_hapmap_exons_phastcons.pkl: hapmap_genetic_map.bed conserved_slop.bed.gz hg38_seqlens.tsv
# 	python ../../bgspy/bgspy.py calcb --ncores 60 --recmap hapmap_genetic_map.bed \
#      --features conserved_slop.bed.gz --seqlens hg19_seqlens.tsv --tg -5:-1:9 \
#      --wg -9:-7:5 --output B_hapmap_exons_phastcons.pkl
