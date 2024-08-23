import csv

from VideoComponents import SRT
from VideoComponents import VideoScene
from VideoComponents import MergedScene
from VideoComponents import TimeRangeUtils
 

def create_string_lines(srts, start_line, token_limit, show_order):
    pos = start_line
    count = 0
    sb = []
    while pos < len(srts) and (token_limit <= 0 or count < token_limit):
        srt = srts[pos]
        if srt.optional:
            pos += 1
            continue
        if token_limit > 0 and srt.content.length + count > token_limit:
            break
        line = f"{srt.order}: " if show_order else ""
        line += srt.content + "\n"
        sb.append(line)
        pos += 1
    return ''.join(sb), pos


def parse_srt_timeline(part):
    pos = part.find(",")
    if pos > 0:
        part = part[:pos]
    fs = part.split(":")
    return int(fs[0]) * 60 * 60 + int(fs[1]) * 60 + int(fs[2])


def load_srt(file):
    srts = []
    try:
        with open(file, "r") as f:
            lines = f.readlines()
            srt = None
            for line in lines:
                read_line = line.strip()
                if len(read_line) == 0:
                    continue
                try:
                    n = int(read_line)
                    srt = SRT()
                    srts.append(srt)
                    srt.order = n
                    continue
                except ValueError:
                    pass
                if srt.start == 0:
                    fs = read_line.split("-->")
                    srt.start = parse_srt_timeline(fs[0].strip())
                    srt.end = parse_srt_timeline(fs[1].strip())
                    continue
                if srt.content is None:
                    srt.content = read_line
                else:
                    srt.content += "" + read_line
                if "<" in read_line and ">" in read_line:
                    srt.optional = True
                if "{\\" in read_line and "}" in read_line:
                    srt.optional = True
    except FileNotFoundError:
        return None
    return srts


def load_video_scenes(file):
    scenes = []
    try:
        with open(file, "r") as f:
            reader = csv.reader(f)
            found = False
            for row in reader:
                read_line = ', '.join(row)
                read_line = read_line.strip()
                if len(read_line) == 0:
                    continue
                if read_line.startswith("Scene Number"):
                    found = True
                    continue
                if not found:
                    continue
                fs = read_line.split(",")
                scene = VideoScene()
                scenes.append(scene)
                scene.order = int(fs[0])
                scene.start = float(fs[3])
                scene.end = float(fs[6])
    except FileNotFoundError:
        return None
    return scenes


def merge(home, filename_no_ext):
    srts = load_srt(f"{home}/{filename_no_ext}.srt")
    scenes = load_video_scenes(f"{home}/{filename_no_ext}-Scenes.csv")
    return merge_scenes(srts, scenes, 20)


def merge_scenes(srts, scenes, maxtimerange):
    print(f"SRT Size = {len(srts)}")
    print(f"VideoScene Size = {len(scenes)}")

    merged = []
    order = 0
    cur_video_scene_pos = 0
    cur_srt_pos = 0
    while cur_video_scene_pos < len(scenes):
        vs = scenes[cur_video_scene_pos]
        m_scene = MergedScene()
        merged.append(m_scene)
        m_scene.merge(vs)
        m_scene.order = order
        order += 1

        cur_video_scene_pos += 1

        while cur_srt_pos < len(srts):
            cur_srt = srts[cur_srt_pos]
            if cur_srt.optional:
                cur_srt_pos += 1
                continue
            if TimeRangeUtils.overlap(m_scene, cur_srt, 2):
                m_scene.mergeSRT(cur_srt)
                if m_scene.end - m_scene.start > maxtimerange and len(m_scene.mergedContent) > 200:
                    break;
                while cur_video_scene_pos < len(scenes):
                    vs2 = scenes[cur_video_scene_pos]
                    if TimeRangeUtils.overlap(vs2, m_scene, 0):
                        m_scene.merge(vs2)
                        cur_video_scene_pos += 1
                        if m_scene.end - m_scene.start > maxtimerange and len(m_scene.mergedContent) > 200:
                            break
                    else:
                        break
            if cur_srt.start > m_scene.end:
                break
            cur_srt_pos += 1

    return merged
