import miditoolkit
import argparse
import os
import re
import inspect
import math
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='')

project_dir = "/screamlab/home/tanch/variable-length-piano-expansion"
parser.add_argument('--result-dir', type=str, default=f"{project_dir}/expand_result/selected")
args = parser.parse_args()

class Song:
    def __init__(self):
        self.idx = None
        self.bd_start = None
        self.bd_end = None
        self.notes = None
        self.tpb = None
        self.ts = None

def main():
    songs = {}
    for song_dir in os.listdir(args.result_dir):
        if re.match("song_[\d]+_bd", song_dir):
            entries = song_dir.split("_")

            song = Song()
            song.idx = int(entries[1])
            song.bd_start = int(entries[-3])-1
            song.bd_end = int(entries[-1])-1

            midi_file = os.path.join(args.result_dir, song_dir, f"song_{song.idx}_result_0.midi")

            midi = miditoolkit.midi.parser.MidiFile(midi_file)
            song.notes = midi.instruments[0].notes
            song.tpb = midi.ticks_per_beat
            song.ts = midi.time_signature_changes[0]
            print(song_dir, song.idx, song.bd_start, song.bd_end, len(song.notes), song.tpb, song.ts)
            songs[song.idx] = song

    # PCHE
    for song in songs.values():
        origin, expanded = PCHE(song)
        #print(song.idx, origin, expanded, expanded-origin)

    #song = songs[2125]
    #song = songs[450]
    song = songs[300]
    #song = songs[3750]
    song = songs[325]
    song = songs[600]
    for i in range(1, 14):
        print(f"{i+1}-{i+2}", RHCE(song, [i-1, i], [i+1, i+2]))

    origin = [i for i in range(0,16)]
    rhce = []
    axis = []
    for o in origin:
        if o == song.bd_start or o == song.bd_end:
            axis.append(f"B{o}")
        else:
            axis.append(f"{o}")
    for i in range(1, len(origin)-2):
        rhce.append(RHCE(song, [origin[i-1], origin[i]], [origin[i+1], origin[i+2]]))
        print(f"{i+1}-{i+2}", RHCE(song, [origin[i-1], origin[i]], [origin[i+1], origin[i+2]]))

    print(axis[1:-2])
    #plt.scatter(axis[1:-2], rhce)
    #plt.show()

    song_list = [(song.idx, song) for song in songs.values()]
    song_list.sort(key=lambda t: t[0])
    song_list = [song for idx, song in song_list]

    song_label = [i*2+1 for i in range(10)]
    font_size = 36

    # plot (C_past vs C_future), (C_past, C*) and (C_future, C*)
    # pf = []
    # ep = []
    # ef = []
    # for song in songs.values():
    #     print(song.idx)
    #     bar_past = [i for i in range(0, song.bd_start)]
    #     bar_expand = [i for i in range(song.bd_start, song.bd_end+1)]
    #     bar_future = [i for i in range(song.bd_end, 16)]
    #     pf.append(PCHCE(song, bar_past, bar_future))
    #     ep.append(PCHCE(song, bar_expand, bar_past))
    #     ef.append(PCHCE(song, bar_expand, bar_future))
    # plt.scatter(range(len(pf)), pf)
    # plt.scatter(range(len(pf)), ep)
    # plt.scatter(range(len(pf)), ef)
    # plt.savefig('fig_pchce.png')

    pf = []
    ep = []
    ef = []
    for song in song_list:
        print(song.idx)
        bar_past = [i for i in range(0, song.bd_start)]
        bar_expand = [i for i in range(song.bd_start, song.bd_end+1)]
        bar_future = [i for i in range(song.bd_end, 16)]
        pf.append(GS(song, bar_past, bar_future))
        ep.append(GS(song, bar_expand, bar_past))
        ef.append(GS(song, bar_expand, bar_future))
    plt.figure(figsize=(16,9))
    plt.subplots_adjust(left=0.16, bottom=0.2)
    plt.rc('font', size=font_size)
    # plt.scatter(range(len(pf)), pf, label="past vs future")
    # plt.scatter(range(len(pf)), ep, label="expanded vs past")
    # plt.scatter(range(len(pf)), ef, label='expanded vs future')
    plt.bar([i+1 for i in range(20)], [ef[i] - ep[i] for i in range(len(pf))], label='future - past')
    # plt.legend(loc='upper right')
    plt.xticks(song_label, rotation=0)
    plt.xlabel("Songs", labelpad=24)
    plt.ylabel("Metric Difference")
    plt.savefig('fig_gs.png')

    print()
    pf = []
    ep = []
    ef = []
    for order, song in enumerate(song_list):
        print(order, song.idx)
        bar_past = [i for i in range(0, song.bd_start)]
        bar_expand = [i for i in range(song.bd_start, song.bd_end+1)]
        bar_future = [i for i in range(song.bd_end, 16)]
        pf.append(RHCE(song, bar_past, bar_future))
        ep.append(RHCE(song, bar_expand, bar_past))
        ef.append(RHCE(song, bar_expand, bar_future))
    plt.figure(figsize=(16, 9))
    plt.subplots_adjust(left=0.12, bottom=0.2)
    plt.rc('font', size=font_size)
    # plt.scatter(range(len(pf)), pf, label="past vs future")
    # plt.scatter(range(len(pf)), ep, label="expanded vs past")
    # plt.scatter(range(len(pf)), ef, label='expanded vs future')
    plt.bar([i+1 for i in range(20)], [ep[i] - ef[i] for i in range(len(pf))], label='-(future - past)')
    plt.xticks(song_label, rotation=0)
    plt.xlabel("Songs", labelpad=24)
    plt.ylabel("Metric Difference", labelpad=24)
    plt.savefig('fig_rhce.png')

def PCHE(song):
    bar_groups = notes_to_bar_group(song.notes, song.tpb, song.ts.numerator, 16)

    # origin
    pcd = [0 for _ in range(12)] # pitch class distribution
    for bar, group in enumerate(bar_groups):
        if song.bd_start <= bar <= song.bd_end:
            continue
        for note in group:
            pitch_class = note.pitch % 12
            pcd[pitch_class] += 1
    origin = entropy(pcd)

    # expanded
    pcd = [0 for _ in range(12)] # pitch class distribution
    for bar, group in enumerate(bar_groups):
        for note in group:
            pitch_class = note.pitch % 12
            pcd[pitch_class] += 1
    expanded = entropy(pcd)

    return origin, expanded

def PCHCE(song, bg1, bg2):
    bar_groups = notes_to_bar_group(song.notes, song.tpb, song.ts.numerator, 16)

    ce = 0
    for i in bg1:
        for j in bg2:
            pcd1 = [1e-4 for _ in range(12)] # pitch class distribution
            pcd2 = [1e-4 for _ in range(12)] # pitch class distribution
            for note in bar_groups[i]:
                pitch_class = note.pitch % 12
                pcd1[pitch_class] += 1
            for note in bar_groups[j]:
                pitch_class = note.pitch % 12
                pcd2[pitch_class] += 1
            ce += cross_entropy(pcd1, pcd2)
            #print(cross_entropy(pcd1, pcd2))
    ce /= (len(bg1) * len(bg2))

    return ce

def GS(song, bg1, bg2):
    bar_groups = notes_to_bar_group(song.notes, song.tpb, song.ts.numerator, 16)

    avg = 0
    for b1 in bg1:
        for b2 in bg2:
            Q = 4*16
            g1 = [0 for _ in range(Q)] # 16 subbeats per beat
            g2 = [0 for _ in range(Q)]

            for note in bar_groups[b1]:
                subbeat = int(note.start / song.tpb % song.ts.numerator // (1/16))
                g1[subbeat] = 1

            for note in bar_groups[b2]:
                subbeat = int(note.start / song.tpb % song.ts.numerator // (1/16))
                g2[subbeat] = 1

            gs = 0
            for i in range(Q):
                gs += g1[i] ^ g2[i]
            avg += 1 - (gs / Q)
    avg = avg / (len(bg1) * len(bg2))

    return avg

def RHCE(song, bg1, bg2): # register histogram cross entropy
    bar_groups = notes_to_bar_group(song.notes, song.tpb, song.ts.numerator, 16)

    ce = 0
    for i in bg1:
        for j in bg2:
            h1 = [1e-4 for _ in range(7)] # 16 subbeats per beat
            h2 = [1e-4 for _ in range(7)]
            for note in bar_groups[i]:
                octave = int((note.pitch - 21) // 12)
                h1[octave] += 1
            for note in bar_groups[j]:
                octave = int((note.pitch - 21) // 12)
                h2[octave] += 1
            ce += cross_entropy(h1, h2)
    ce /= (len(bg1) * len(bg2))
    return ce

def entropy(v):
    v = normalize(v)

    entropy = 0
    for p in v:
        if p > 0:
            entropy += p * math.log(p, 2)
    return -entropy

def cross_entropy(p, q):
    p = normalize(p)
    q = normalize(q)

    s = 0
    for i in range(len(p)):
        s += p[i] * -math.log(q[i], 2)
    return s

def normalize(v):
    s = 0
    for p in v:
        s += p
    for i, _ in enumerate(v):
        v[i] /= s
    return v

def mean(v):
    s = 0
    for p in v:
        s += p
    return s / len(v)

def min_pos(v):
    for p in v:
        if p > 0:
            m = p
    for p in v:
        if p != 0 and p < m:
            m = p
    return m

def notes_to_bar_group(notes, tpb, beat_per_bar, max_bar):
    groups = [[] for _ in range(max_bar)]
    for note in notes:
        bar = int(note.start / tpb // beat_per_bar)
        groups[bar].append(note)
    return groups

if __name__ == '__main__':
    main()
