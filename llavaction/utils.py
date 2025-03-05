import datetime
import logging
import logging.handlers
import os
import sys
import numpy as np

import requests

from llavaction.constants import LOGDIR

server_error_msg = "**NETWORK ERROR DUE TO HIGH TRAFFIC. PLEASE REGENERATE OR REFRESH THIS PAGE.**"
moderation_msg = "I am sorry. Your input may violate our content moderation guidelines. Please avoid using harmful or offensive content."

handler = None

import torch.distributed as dist

try:
    import av
    from decord import VideoReader, cpu
except ImportError:
    print("Please install pyav to use video processing functions.")

def get_frame_ids(start_frame, end_frame, num_segments=32, jitter=True):
    frame_ids = np.convolve(np.linspace(start_frame, end_frame, num_segments + 1), [0.5, 0.5], mode='valid')
    if jitter:
        seg_size = float(end_frame - start_frame - 1) / num_segments
        shift = (np.random.rand(num_segments) - 0.5) * seg_size
        frame_ids += shift
    return frame_ids.astype(int).tolist()

# def get_video_reader(videoname, num_threads, fast_rrc, rrc_params, fast_rcc, rcc_params):
#     video_reader = None
#     if fast_rrc:
#         video_reader = VideoReader(
#             videoname,
#             num_threads=num_threads,
#             width=rrc_params[0], height=rrc_params[0],
#             use_rrc=True, scale_min=rrc_params[1][0], scale_max=rrc_params[1][1],
#         )
#     elif fast_rcc:
#         video_reader = VideoReader(
#             videoname,
#             num_threads=num_threads,
#             width=rcc_params[0], height=rcc_params[0],
#             use_rcc=True,
#         )
#     else:
#         video_reader = VideoReader(videoname, num_threads=num_threads)
#     return video_reader

# def video_loader(root, vid, ext, second, end_second,
#                  chunk_len=300, fps=30, clip_length=32,
#                  threads=1,
#                  fast_rrc=False, rrc_params=(224, (0.5, 1.0)),
#                  fast_rcc=False, rcc_params=(224, ),
#                  jitter=False):
#     assert fps > 0, 'fps should be greater than 0'

#     if chunk_len == -1:
#         vr = get_video_reader(
#             osp.join(root, '{}.{}'.format(vid, ext)),
#             num_threads=threads,
#             fast_rrc=fast_rrc, rrc_params=rrc_params,
#             fast_rcc=fast_rcc, rcc_params=rcc_params,
#         )
#         end_second = min(end_second, len(vr) / fps)

#         # calculate frame_ids
#         frame_offset = int(np.round(second * fps))
#         total_duration = max(int((end_second - second) * fps), clip_length)
#         frame_ids = get_frame_ids(frame_offset, min(frame_offset + total_duration, len(vr)), num_segments=clip_length, jitter=jitter)

#         # load frames
#         assert max(frame_ids) < len(vr)
#         try:
#             frames = vr.get_batch(frame_ids).asnumpy()
#         except decord.DECORDError as error:
#             print(error)
#             frames = vr.get_batch([0] * len(frame_ids)).asnumpy()
    
#         return torch.from_numpy(frames.astype(np.float32))

#     else:
#         chunk_start = int(second) // chunk_len * chunk_len
#         chunk_end = int(end_second) // chunk_len * chunk_len
#         while True:
#             video_filename = osp.join(root, '{}.{}'.format(vid, ext), '{}.{}'.format(chunk_end, ext))
#             if not osp.exists(video_filename):
#                 # print("{} does not exists!".format(video_filename))
#                 chunk_end -= chunk_len
#             else:
#                 vr = decord.VideoReader(video_filename)
#                 end_second = min(end_second, (len(vr) - 1) / fps + chunk_end)
#                 assert chunk_start <= chunk_end
#                 break
#         # calculate frame_ids
#         frame_ids = get_frame_ids(
#             int(np.round(second * fps)),
#             int(np.round(end_second * fps)),
#             num_segments=clip_length, jitter=jitter
#         )
#         all_frames = []
#         # allocate absolute frame-ids into the relative ones
#         for chunk in range(chunk_start, chunk_end + chunk_len, chunk_len):
#             rel_frame_ids = list(filter(lambda x: int(chunk * fps) <= x < int((chunk + chunk_len) * fps), frame_ids))
#             rel_frame_ids = [int(frame_id - chunk * fps) for frame_id in rel_frame_ids]
#             vr = get_video_reader(
#                 osp.join(root, '{}.{}'.format(vid, ext), '{}.{}'.format(chunk, ext)),
#                 num_threads=threads,
#                 fast_rrc=fast_rrc, rrc_params=rrc_params,
#                 fast_rcc=fast_rcc, rcc_params=rcc_params,
#             )
#             try:
#                 frames = vr.get_batch(rel_frame_ids).asnumpy()
#             except decord.DECORDError as error:
#                 print(error)
#                 frames = vr.get_batch([0] * len(rel_frame_ids)).asnumpy()
#             except IndexError:
#                 print(root, vid, ext, second, end_second)
#             all_frames.append(frames)
#             if sum(map(lambda x: x.shape[0], all_frames)) == clip_length:
#                 break
#         res = torch.from_numpy(np.concatenate(all_frames, axis=0).astype(np.float32))
#         assert res.shape[0] == clip_length, "{}, {}, {}, {}, {}, {}, {}".format(root, vid, second, end_second, res.shape[0], rel_frame_ids, frame_ids)
#         return res

def process_EK100_video_with_decord(video_file, data_args, start_second, end_second, chunk_len):
    fps = 30
    start_frame = int(start_second * fps)
    end_frame = int(end_second * fps)
    chunk_start = int(start_second) // chunk_len * chunk_len
    chunk_end = int(end_second) // chunk_len * chunk_len
    video_time = end_second - start_second
    while True:
        video_filename = os.path.join(video_file, '{}.MP4'.format(chunk_end))
        if not os.path.exists(video_filename):
            # print("{} does not exists!".format(video_filename))
            chunk_end -= chunk_len
        else:
            vr = VideoReader(video_filename, ctx=cpu(0), num_threads=1)
            end_second = min(end_second, (len(vr) - 1) / fps + chunk_end)
            assert chunk_start <= chunk_end
            break
    
    # calculate frame_ids
    frame_ids = get_frame_ids(start_frame, end_frame, num_segments=data_args.frames_upbound, jitter=False)
  
    
    
    all_frames = []
    all_frame_ids = []
    # allocate absolute frame-ids into the relative ones
    for chunk in range(chunk_start, chunk_end + chunk_len, chunk_len):
        rel_frame_ids = list(filter(lambda x: int(chunk * fps) <= x < int((chunk + chunk_len) * fps), frame_ids))
        rel_frame_ids = [int(frame_id - chunk * fps) for frame_id in rel_frame_ids]
        vr = VideoReader(os.path.join(video_file, '{}.MP4'.format(chunk)),ctx=cpu(0), num_threads=1)
        frames = vr.get_batch(rel_frame_ids).asnumpy()
        all_frames.append(frames)
        all_frame_ids.append(frame_ids)
        vr.seek(0)
        if sum(map(lambda x: x.shape[0], all_frames)) == data_args.frames_upbound:
            break

    video = np.concatenate(all_frames, axis=0).astype(np.float32)

    all_frame_ids = np.concatenate(all_frame_ids, axis = 0)
    frame_time = [e/fps for e in all_frame_ids]
    frame_time-= frame_time[0]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

    num_frames_to_sample = len(frame_ids)

    return video, video_time, frame_time, num_frames_to_sample

def process_video_with_decord(video_file, data_args):
    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    avg_fps = round(vr.get_avg_fps() / data_args.video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]
    frame_time = [i/avg_fps for i in frame_idx]

    
    if data_args.frames_upbound > 0:
        if len(frame_idx) > data_args.frames_upbound or data_args.force_sample:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    
    video = vr.get_batch(frame_idx).asnumpy()
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])

    num_frames_to_sample = num_frames = len(frame_idx)
    # https://github.com/dmlc/decord/issues/208
    vr.seek(0)
    return video, video_time, frame_time, num_frames_to_sample

def process_video_with_pyav(video_file, data_args):
    container = av.open(video_file)
    # !!! This is the only difference. Using auto threading
    container.streams.video[0].thread_type = "AUTO"

    video_frames = []
    for packet in container.demux():
        if packet.stream.type == 'video':
            for frame in packet.decode():
                video_frames.append(frame)
    total_frame_num = len(video_frames)
    video_time = video_frames[-1].time
    avg_fps = round(total_frame_num / video_time / data_args.video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]

    if data_args.frames_upbound > 0:
        if len(frame_idx) > data_args.frames_upbound:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()


    frames = [video_frames[i] for i in frame_idx]
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def rank_print(*args):
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)

def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(filename, when="D", utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ""


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        print(f"######################### Moderation Error: {e} #########################")
        flagged = False
    except KeyError as e:
        print(f"######################### Moderation Error: {e} #########################")
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"
