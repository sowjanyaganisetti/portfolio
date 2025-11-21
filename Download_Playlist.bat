@echo off
yt-dlp -o "%(playlist_index)s - %(title)s.%(ext)s" https://www.youtube.com/playlist?list=PL5x4-s2in7NX53L3shKYABz00F2Tld6Vf
pause