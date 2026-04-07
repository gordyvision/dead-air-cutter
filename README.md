# dead-air-cutter
Using a simple python script combined with AI to cut out dead air in your videos (works with game playthroughs as well)

I made this for myself because I used to spend a lot of time cutting out "dead air" in my playthroughs. Really cut down my editing time. Possibly someone else will find this useful too. Parameters can be tweaked as needed.

Personally this is the command I use:
python .\dead-air-cutter.py .\some-long-playthrough.mp4 long-playthrough-aicut.mp4 --pre-roll 0.25 --post-rol 0.25 --min-segment-len 0.4 --min-gap-len 0.4 --vad-threshold 0.5

Sometimes I still end up cutting out more space between words, just depends though. Obviously the script leaves in any "uhhh" and "umm" but that's to be expected.


