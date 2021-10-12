import os
import sys
import argparse
import re
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('-src_dir', type=str, default=None)
parser.add_argument('-dist_dir', type=str, default=None)
parser.add_argument('-index_file', type=str, default='./content.html')
parser.add_argument('-result_file', type=str, default='./result.txt')
args = parser.parse_args()
print(args.src_dir)
print(args.dist_dir)
print(args.index_file)
print(args.result_file)

def main():
    on_bd, not_on_bd, same_good, same_bad, all_files = read_result_file(args.result_file)
    if on_bd[0] == "":
        on_bd = []
    if not_on_bd[0] == "":
        not_on_bd = []
    if same_good[0] == "":
        same_good = []
    if same_bad[0] == "":
        same_bad = []
    on_bd_content = []
    not_on_bd_content = []
    same_good_content = []
    same_bad_content = []
    all_files_content = []

    # clean up the assets folder
    for old_file in os.listdir(args.dist_dir):
        if re.search(".midi", old_file):
            os.remove(os.path.join(args.dist_dir, old_file))
            print("remove:", os.path.join(args.dist_dir, old_file))

    print("all files")
    for song_idx in all_files:
        midi_origin = os.path.join(
            args.src_dir,
            song_dir(song_idx, on_bd=True),
            f'song_{song_idx}_origin.midi')
        midi_bd = os.path.join(
            args.src_dir,
            song_dir(song_idx, on_bd=True),
            f'song_{song_idx}_result_0.midi')
        midi_notbd = os.path.join(
            args.src_dir,
            song_dir(song_idx, on_bd=False),
            f'song_{song_idx}_result_0.midi')

        # copy midi into assets folder
        shutil.copy2(midi_origin, os.path.join(args.dist_dir, f"song_{song_idx}_origin.midi"))
        shutil.copy2(midi_bd, os.path.join(args.dist_dir, f"song_{song_idx}_bd.midi"))
        shutil.copy2(midi_notbd, os.path.join(args.dist_dir, f"song_{song_idx}_notbd.midi"))
        print("copy:", os.path.join(args.dist_dir, f"song_{song_idx}_origin.midi"))
        print("copy:", os.path.join(args.dist_dir, f"song_{song_idx}_bd.midi"))
        print("copy:", os.path.join(args.dist_dir, f"song_{song_idx}_notbd.midi"))

        # generate content block
        all_files_content.append(content_block_from_template(song_idx))

    """
    print("on boundary")
    for song_idx in on_bd:
        midi_origin = os.path.join(
            args.src_dir,
            song_dir(song_idx, on_bd=True),
            f'song_{song_idx}_origin.midi')
        midi_bd = os.path.join(
            args.src_dir,
            song_dir(song_idx, on_bd=True),
            f'song_{song_idx}_result_0.midi')
        midi_notbd = os.path.join(
            args.src_dir,
            song_dir(song_idx, on_bd=False),
            f'song_{song_idx}_result_0.midi')

        # copy midi into assets folder
        shutil.copy2(midi_origin, os.path.join(args.dist_dir, f"song_{song_idx}_origin.midi"))
        shutil.copy2(midi_bd, os.path.join(args.dist_dir, f"song_{song_idx}_bd.midi"))
        shutil.copy2(midi_notbd, os.path.join(args.dist_dir, f"song_{song_idx}_notbd.midi"))
        print("copy:", os.path.join(args.dist_dir, f"song_{song_idx}_origin.midi"))
        print("copy:", os.path.join(args.dist_dir, f"song_{song_idx}_bd.midi"))
        print("copy:", os.path.join(args.dist_dir, f"song_{song_idx}_notbd.midi"))

        # generate content block
        on_bd_content.append(content_block_from_template(song_idx))

    print("not on boundary")
    for song_idx in not_on_bd:
        midi_origin = os.path.join(
            args.src_dir,
            song_dir(song_idx, on_bd=True),
            f'song_{song_idx}_origin.midi')
        midi_bd = os.path.join(
            args.src_dir,
            song_dir(song_idx, on_bd=True),
            f'song_{song_idx}_result_0.midi')
        midi_notbd = os.path.join(
            args.src_dir,
            song_dir(song_idx, on_bd=False),
            f'song_{song_idx}_result_0.midi')

        # copy midi into assets folder
        shutil.copy2(midi_origin, os.path.join(args.dist_dir, f"song_{song_idx}_origin.midi"))
        shutil.copy2(midi_bd, os.path.join(args.dist_dir, f"song_{song_idx}_bd.midi"))
        shutil.copy2(midi_notbd, os.path.join(args.dist_dir, f"song_{song_idx}_notbd.midi"))
        print("copy:", os.path.join(args.dist_dir, f"song_{song_idx}_origin.midi"))
        print("copy:", os.path.join(args.dist_dir, f"song_{song_idx}_bd.midi"))
        print("copy:", os.path.join(args.dist_dir, f"song_{song_idx}_notbd.midi"))

        # generate content block
        not_on_bd_content.append(content_block_from_template(song_idx))

    print("same good")
    for song_idx in same_good:
        midi_origin = os.path.join(
            args.src_dir,
            song_dir(song_idx, on_bd=True),
            f'song_{song_idx}_origin.midi')
        midi_bd = os.path.join(
            args.src_dir,
            song_dir(song_idx, on_bd=True),
            f'song_{song_idx}_result_0.midi')
        midi_notbd = os.path.join(
            args.src_dir,
            song_dir(song_idx, on_bd=False),
            f'song_{song_idx}_result_0.midi')

        # copy midi into assets folder
        shutil.copy2(midi_origin, os.path.join(args.dist_dir, f"song_{song_idx}_origin.midi"))
        shutil.copy2(midi_bd, os.path.join(args.dist_dir, f"song_{song_idx}_bd.midi"))
        shutil.copy2(midi_notbd, os.path.join(args.dist_dir, f"song_{song_idx}_notbd.midi"))
        print("copy:", os.path.join(args.dist_dir, f"song_{song_idx}_origin.midi"))
        print("copy:", os.path.join(args.dist_dir, f"song_{song_idx}_bd.midi"))
        print("copy:", os.path.join(args.dist_dir, f"song_{song_idx}_notbd.midi"))

        # generate content block
        same_good_content.append(content_block_from_template(song_idx))

    print("same bad")
    for song_idx in same_bad:
        midi_origin = os.path.join(
            args.src_dir,
            song_dir(song_idx, on_bd=True),
            f'song_{song_idx}_origin.midi')
        midi_bd = os.path.join(
            args.src_dir,
            song_dir(song_idx, on_bd=True),
            f'song_{song_idx}_result_0.midi')
        midi_notbd = os.path.join(
            args.src_dir,
            song_dir(song_idx, on_bd=False),
            f'song_{song_idx}_result_0.midi')

        # copy midi into assets folder
        shutil.copy2(midi_origin, os.path.join(args.dist_dir, f"song_{song_idx}_origin.midi"))
        shutil.copy2(midi_bd, os.path.join(args.dist_dir, f"song_{song_idx}_bd.midi"))
        shutil.copy2(midi_notbd, os.path.join(args.dist_dir, f"song_{song_idx}_notbd.midi"))
        print("copy:", os.path.join(args.dist_dir, f"song_{song_idx}_origin.midi"))
        print("copy:", os.path.join(args.dist_dir, f"song_{song_idx}_bd.midi"))
        print("copy:", os.path.join(args.dist_dir, f"song_{song_idx}_notbd.midi"))

        # generate content block
        same_bad_content.append(content_block_from_template(song_idx))
    """

    # write context into html file
    with open(args.index_file, 'w') as f:
        f.write(index_with_content(
            "".join(on_bd_content),
            "".join(not_on_bd_content),
            "".join(same_good_content),
            "".join(same_bad_content),
            "".join(all_files_content),
        ))
        print("generate index.html:", args.index_file)

def read_result_file(file):
    with open(file) as f:
        on_bd_line = f.readline().strip()
        not_on_bd_line = f.readline().strip()
        same_good_line = f.readline().strip()
        same_bad_line = f.readline().strip()
        all_file_line = f.readline().strip()

        on_bd = on_bd_line.split(':')[1].split(',')
        not_on_bd = not_on_bd_line.split(':')[1].split(',')
        same_good = same_good_line.split(':')[1].split(',')
        same_bad = same_bad_line.split(':')[1].split(',')
        all_files = all_file_line.split(':')[1].split(',')
    return on_bd, not_on_bd, same_good, same_bad, all_files

def song_dir(song_idx, on_bd: bool):
    if not hasattr(song_dir, 'song_dir_dict'):
        song_dirs = os.listdir(args.src_dir)
        song_dir.song_dir_dict = \
            {"_".join(d.split('_')[1:3]): d for d in song_dirs}
    if on_bd:
        return song_dir.song_dir_dict["_".join([song_idx, "bd"])]
    else:
        return song_dir.song_dir_dict["_".join([song_idx, "notbd"])]

song_absolute_id = 0
def content_block_from_template(song_idx):
    global song_absolute_id
    #song_idx = f"{song_idx:02d}"
    dir_url = "https://cdn.jsdelivr.net/gh/tanchihpin0517/variable-length-piano-expansion/docs/assets/songs/expansion"
    #dir_url = "http://screamviolin.csie.ncku.edu.tw:8000/assets/songs/expansion"
    bd_url = f"{dir_url}/song_{song_idx}_bd.midi"
    origin_url = f"{dir_url}/song_{song_idx}_origin.midi"
    song_absolute_id += 1
    return \
f"""
<section id="inpainted-music-song-{song_idx}">
  <h3>Song {song_absolute_id}</h3>
  <div class='anchor-container'>
    <a class='selected'
      midi-url="{origin_url}">Origin</a>
    <a
      midi-url="{bd_url}">Expended</a>
  </div>
  <midi-visualizer id='inpainted-music-song-{song_idx}-visualizer'
    src="{origin_url}">
  </midi-visualizer>
  <midi-player id='inpainted-music-song-{song_idx}-player'
    src="{origin_url}"
    sound-font="https://storage.googleapis.com/magentadata/js/soundfonts/salamander"
    visualizer="#inpainted-music-song-{song_idx}-visualizer">
  </midi-player>
</section>
"""

#def content_block_from_template(song_idx):
#    #song_idx = f"{song_idx:02d}"
#    dir_url = "https://cdn.jsdelivr.net/gh/tanchihpin0517/variable-length-piano-expansion/docs/assets/songs/expansion"
#    #dir_url = "http://screamviolin.csie.ncku.edu.tw:8000/assets/songs/expansion"
#    bd_url = f"{dir_url}/song_{song_idx}_bd.midi"
#    notbd_url = f"{dir_url}/song_{song_idx}_notbd.midi"
#    origin_url = f"{dir_url}/song_{song_idx}_origin.midi"
#    return \
#f"""
#<section id="inpainted-music-song-{song_idx}">
#  <h3>Song {song_idx}</h3>
#  <div class='anchor-container'>
#    <a class='selected'
#      midi-url="{origin_url}">Origin</a>
#    <a
#      midi-url="{bd_url}">On Boundary</a>
#    <a
#      midi-url="{notbd_url}">Not On Boundary</a>
#  </div>
#  <midi-visualizer id='inpainted-music-song-{song_idx}-visualizer'
#    src="{origin_url}">
#  </midi-visualizer>
#  <midi-player id='inpainted-music-song-{song_idx}-player'
#    src="{origin_url}"
#    sound-font="https://storage.googleapis.com/magentadata/js/soundfonts/salamander"
#    visualizer="#inpainted-music-song-{song_idx}-visualizer">
#  </midi-player>
#</section>
#"""

def index_with_content(on_bd, not_on_bd, same_good, same_bad, all_files):
    return \
f"""
<!DOCTYPE html>
<html lang="en">

<head>
  <meta charset="UTF-8">
  <meta http-equiv="X-UA-Compatible" content="IE=edge">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Infilling</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Arimo&family=Varela+Round&display=swap" rel="stylesheet">
  <link href="https://fonts.googleapis.com/css2?family=Literata&display=swap" rel="stylesheet">
  <link rel="stylesheet" href="index.css">
  <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
  <script src="./assets/js/index.js"></script>
</head>

<body>
  <header class="page-header">
    <h1 class="">Music Score Expansion with Variable-Length Infilling</h1>
    <h2 class="">Demo of music score expansion with the model proposed in "Variable-Length Music Score Infilling via XLNet and Musically Specialized Positional Encoding"</h2>
    <a class="page-header__link" target='_blank' href="https://github.com/tanchihpin0517/variable-length-piano-expansion">
      Github</a>
    <a class="page-header__link" target='_blank' href="">
      Paper</a>
  </header>

  <main>
    <section id="inpainted-music">
      <h2>Music expanded by the variable-length infilling model</h2>
      <p>These songs are the original and expended music in our work.</p>
      <p>Note: If you are browsing this page using an iPhone, please remember to turn the silent mode off through the
        switch on the side of your iPhone.</p>

      {all_files}
    </section>

    <footer class="site-footer">
      <span class="site-footer-credit">We use
        <a target='_blank' href='https://github.com/cifkao/html-midi-player'>html_midi_player</a>
        built and kindly shared publicly by
        <a target='_blank' href="https://ondrej.cifka.com">Ondřej Cífka</a>
        for visualizing the piano rolls of the MIDI files. We have also made this implementation
        of the website public at the <a target='_blank'
          href="https://github.com/jackyhsiung/piano-infilling-demo">Github repository</a>.
      </span>
    </footer>
  </main>
  <script
    src="https://cdn.jsdelivr.net/combine/npm/tone@14.7.58,npm/@magenta/music@1.21.0/es6/core.js,npm/focus-visible@5,npm/html-midi-player@1.1.0"></script>

</body>

</html>
"""

if __name__ == '__main__':
    main()
