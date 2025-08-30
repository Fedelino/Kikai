import subprocess
import numpy as np 
import os
from gpmfstream import Stream
from tqdm import tqdm
from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as plt
from reconstruction_utils import get_legend

def get_video_length(filename):
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)

def extract_frames_and_gopro_gravity_vector(video_names, timestamps, width, height, fps, tmp_dir, reverse=False):
    """As input, takes a list of video names of the form: [<video_name1>, <video_name2>], 
    and a list of (minute:second)-timestamps of the example form ["<seconds>-end","begin-<seconds>"]
    using FFMPEG, extracts frames at <fps> frames per second, with the height and width set accordingly,
    and in reverse for videos where the camera moves backwards.
    """

    os.makedirs(tmp_dir + "/rgb", exist_ok=True)
    
    gravity_vectors = []
    total_frames = 0

    for video_id, (video_name, timestamp) in enumerate(zip(video_names, timestamps)):
        
        targetpath = tmp_dir + "/" + video_name.split("/")[-1].split(".")[0].replace(" ", "_").replace("/", "_")
        os.makedirs(targetpath, exist_ok=True)
        
        begin, end = timestamp.split("-")
        ss = ""
        to = ""
        if begin != "begin":
            ss += " -ss " + begin
        if end != "end":
            to += " -to " + end
        
        reverse_flag = ""
        if reverse:
            reverse_flag = "reverse,"
        
        # First: cut video
        #os.system("ffmpeg-7.0.2-amd64-static/ffmpeg -hide_banner -loglevel error"+ss+" " +to+" -y -i '"+video_name+"'  -c copy "+tmp_dir+"/"+str(video_id)+".mp4")
        os.system("ffmpeg -hide_banner -loglevel error"+ss+" " +to+" -y -i '"+video_name+"'  -c copy "+tmp_dir+"/"+str(video_id)+".mp4")
        # Second: scale video to right dimensions
        os.system("ffmpeg -hide_banner -loglevel error  -y -i "+tmp_dir+"/"+str(video_id)+".mp4 -vf scale="+str(width)+":"+str(height)+" "+tmp_dir+"/"+str(video_id)+"_.mp4")
        # Third: extract frames
        os.system("ffmpeg -hide_banner -loglevel error -y -i "+tmp_dir+"/"+str(video_id)+"_.mp4 -vf "+reverse_flag+"fps="+str(fps)+" -qscale:v 2 "+targetpath +"/%07d.jpg")
        
        num_frames = len(os.listdir(targetpath))
        
        for frame in os.listdir(targetpath):
            frameid = int(frame.split(".")[0])
            os.system("mv "+targetpath + "/" + frame + " "+tmp_dir+"/rgb/" + str(frameid + total_frames).zfill(7) + ".jpg")

        total_frames += num_frames
        gravity_vectors.append(get_gravity_vectors(video_name, timestamp, num_frames))
    if gravity_vectors[0] is None:
        return None
    return np.concatenate(gravity_vectors)


def get_gravity_vectors(video, timestamp, number_of_frames):
    """Uses gpmfstream to extract gravity vectors from an MP4 video file."""
    try:
        grav = Stream.extract_streams(video)["GRAV"].data
    except Exception as e:
        print("WARNING: Could not extract gravity vectors from video file:", video,  " is your video an unedited GoPro video?")
        return None    
    length = get_video_length(video)

    begin, end = timestamp.split("-")
    #TODO: timestamp parsing!
    
    if begin == "begin":
        begin = 0
    if end == "end":
        end = length
    begin = float(begin)
    end = float(end)

    grav = grav[int(begin/length*len(grav)):int(end/length*len(grav))]
    inds = np.linspace(0, len(grav)-1,  number_of_frames).astype(np.int32)
    grav = grav[inds]
    grav /= np.linalg.norm(grav, axis=1).reshape(-1, 1)
    return grav

def render_video(img_list, depths, semantic_segmentation, results_npy, fps, class_to_label, label_to_color, tmp_dir, reverse):
    """Renders a 4-panel video (RGB, RGB+Depth, RGB+Seg, 2D map) with size-safe overlays."""
    import os
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    import cv2

    os.makedirs(os.path.join(tmp_dir, "render"), exist_ok=True)

    # ----- Stable depth normalization (global per-clip) -----
    # depths: list/array of (H0,W0); keep a copy for masking, etc.
    depths = np.asarray(depths)  # (N,H0,W0)
    q2, q98 = np.nanquantile(depths, [0.02, 0.98])
    depths = np.nan_to_num(depths, nan=q2)
    depths = np.clip(depths, q2, q98)
    # perceptual tweak
    depths = np.sqrt(depths)
    # normalize to 0..1 using global clip (avoid per-frame blowout)
    d_min = depths.min()
    d_max = depths.max()
    depths_norm = (depths - d_min) / (d_max - d_min + 1e-8)  # (N,H0,W0), 0..1

    # ----- Legend / colors for segmentation & 2D map -----
    class_to_color = {cls_name: label_to_color[cls_label] for cls_name, cls_label in class_to_label.items()}
    legend = get_legend(class_to_color, tmp_dir)  # float 0..1, shape ~ (h,w,3)

    # 2D map inputs
    final_rgb       = results_npy[:, :, 1:4]   # (Hmap, Wmap, 3), uint8
    final_class_rgb = results_npy[:, :, 6:9]   # (Hmap, Wmap, 3), uint8
    frame_index     = results_npy[:, :, 9:10].astype(np.int16)  # (Hmap, Wmap, 1)

    # utility: blend two uint8 RGB images with weights a,b (expects same size)
    def blend_uint8(a, b, wa, wb):
        af = a.astype(np.float32) / 255.0
        bf = b.astype(np.float32) / 255.0
        out = wa * af + wb * bf
        return (np.clip(out, 0, 1) * 255).astype(np.uint8)

    # Prepare legend width relative to the map panel (we'll resize once we know display size)
    legend_u8 = (np.clip(legend, 0, 1) * 255).astype(np.uint8)

    N = len(depths_norm)
    for i in tqdm(range(N)):
        # ---------- Load RGB display frame (we use 640x384 for display) ----------
        rgb = np.array(Image.open(img_list[i]).resize((640, 384)))  # uint8, (H,W,3)
        H, W = rgb.shape[:2]

        # ---------- Build segmentation color overlay ----------
        # map labels to colors
        seg = semantic_segmentation[i]  # (H0,W0) label ids
        color_semseg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for class_name, class_label in class_to_label.items():
            color_semseg[seg == class_label] = label_to_color[class_label]
        # resize segmentation to display size
        seg_u8 = cv2.resize(color_semseg, (W, H), interpolation=cv2.INTER_NEAREST)
        seg_panel = blend_uint8(rgb, seg_u8, 0.3, 0.7)  # (H,W,3) uint8

        # ---------- Mask fish/human depth to 0 before visualization ----------
        d = depths_norm[i].copy()  # (H0,W0), float 0..1
        if 'fish' in class_to_label:
            d[seg == class_to_label['fish']] = 0.0
        if 'human' in class_to_label:
            d[seg == class_to_label['human']] = 0.0

        # match depth to display size
        if d.shape != (H, W):
            d = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)  # (H,W) float 0..1

        # colorize depth and blend with RGB
        depth_rgb = (plt.cm.seismic(d)[..., :3] * 255).astype(np.uint8)  # (H,W,3) uint8
        depth_overlay = blend_uint8(rgb, depth_rgb, 0.2, 0.8)            # (H,W,3) uint8

        # ---------- Build the rolling 2D map panel ----------
        # indicator for frames "seen" so far
        if reverse:
            ind = (frame_index >= i).astype(np.uint8)
        else:
            ind = (frame_index <= i).astype(np.uint8)

        results_rgb      = (final_rgb * ind).astype(np.uint8)       # (Hmap,Wmap,3)
        results_classrgb = (final_class_rgb * ind).astype(np.uint8) # (Hmap,Wmap,3)

        # stack map + class map either vertically or horizontally depending on aspect
        if results_rgb.shape[0] < results_rgb.shape[1]:
            map_stack = np.concatenate([results_rgb, results_classrgb], axis=0)
        else:
            map_stack = np.concatenate([results_rgb, results_classrgb], axis=1)

        # prepend legend on the left (scale legend height to map height)
        # first ensure legend height matches map height
        if i == 0:
            # scale legend to ~10% of map height (tweak if you like)
            target_legend_h = max(32, int(0.1 * map_stack.shape[0]))
            scale = target_legend_h / legend_u8.shape[0]
            legend_u8 = cv2.resize(legend_u8, (int(legend_u8.shape[1] * scale), target_legend_h), interpolation=cv2.INTER_LINEAR)

        # pad/crop legend height to map height
        if legend_u8.shape[0] != map_stack.shape[0]:
            legend_u8 = cv2.resize(legend_u8, (legend_u8.shape[1], map_stack.shape[0]), interpolation=cv2.INTER_LINEAR)

        map_with_legend = np.concatenate([legend_u8, map_stack], axis=1)  # (Hmap, Wlegend+Wmap, 3)

        # finally, resize 2D map panel to match display (H,W)
        map_panel = cv2.resize(map_with_legend, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.uint8)

        # ---------- Compose 2×2 panel ----------
        top    = np.concatenate([rgb,       seg_panel],     axis=1)  # (H, 2W, 3)
        bottom = np.concatenate([depth_overlay, map_panel], axis=1)  # (H, 2W, 3)
        image  = np.concatenate([top, bottom], axis=0)               # (2H, 2W, 3)

        # ---------- Save frame ----------
        fname = f"{(N + 1 - i):07d}.jpg" if reverse else f"{i:07d}.jpg"
        plt.imsave(os.path.join(tmp_dir, "render", fname), image)

    # ---------- Encode video ----------
    os.system(
        "ffmpeg -hide_banner -loglevel error -framerate {fps} -pattern_type glob "
        "-i '{frames}' -c:v libx264 -pix_fmt yuv420p {out}".format(
            fps=fps,
            frames=os.path.join(tmp_dir, "render", "*.jpg"),
            out=os.path.join(tmp_dir, "out.mp4"),
        )
    )