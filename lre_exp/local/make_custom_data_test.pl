#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter
#
# Copyright 2014  David Snyder  Daniel Povey
 


if (@ARGV != 2) {
  print STDERR "Usage: $0 <path-to-LDC2006S31> <output-dir> provide appropriate argument\n";
  exit(1);
}

($base, $out_base_dir) = @ARGV;

$db_file = $base . "/Data_key.txt";
open(DB, "<$db_file")
  || die "Failed opening input file $db_file";

$out_dir = $out_base_dir;
#$data_dir = $base;
if (system("mkdir -p $out_dir") != 0) {
  die "Error making directory $out_dir"; 
}

open(WAV, ">$out_dir" . '/wav.scp') 
  || die "Failed opening output file $out_dir/wav.scp";
open(UTT2LANG, ">$out_dir" . '/utt2lang') 
  || die "Failed opening output file $out_dir/utt2lang";
open(UTT2SPK, ">$out_dir" . '/utt2spk') 
  || die "Failed opening output file $out_dir/utt2spk";
#open(SPK2UTT, ">$out_dir" . '/spk2utt')
  #|| die "Failed opening output file $out_dir/spk2utt";

while($line = <DB>) {
  chomp($line);
  @toks = split(" ", $line);
  $seg_id = lc $toks[0];
  $lang = lc $toks[1];
  $wav = $toks[2];
  #$channel = $toks[3];
  #$duration = $toks[4];
  #$gender = lc $toks[6];
  #$channel = substr($channel, 1, 1); # they are either A1 or B2: we want the
                                     # numeric channel.

  #$wav = "$base/data/lid03e1/test/$duration/$seg_id.sph";
  if (! -f $wav) {
    print STDERR "No such file $wav\n";
    next;
  }
  $uttId = "${lang}-${seg_id}";
  
  print WAV "$uttId $wav\n";
  print UTT2SPK "$uttId $lang\n";
  print UTT2LANG "$uttId $lang\n";
  #print SPK2UTT "$uttId $uttId\n";
}

close(WAV) || die;
close(UTT2SPK) || die;
close(UTT2LANG) || die;
#close(SPK2UTT) || die;
close(DB) || die;

system("utils/fix_data_dir.sh $out_dir");
(system("utils/validate_data_dir.sh --no-text --no-feats $out_dir") == 0) 
  || die "Error validating data dir.";
