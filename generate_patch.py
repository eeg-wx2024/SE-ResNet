import glob
import platform
from tqdm import tqdm
from scipy.interpolate import griddata
from joblib import Parallel, delayed
import pickle
import numpy as np
import mne

parallel_jobs = 1

exclude = ["EXG1", "EXG2", "EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8"]
stim_channel = "Status"
classes = {
    "n02106662": 0,
    "n02124075": 1,
    "n02281787": 2,
    "n02389026": 3,
    "n02492035": 4,
    "n02504458": 5,
    "n02510455": 6,
    "n02607072": 7,
    "n02690373": 8,
    "n02906734": 9,
    "n02951358": 10,
    "n02992529": 11,
    "n03063599": 12,
    "n03100240": 13,
    "n03180011": 14,
    "n03272010": 15,
    "n03272562": 16,
    "n03297495": 17,
    "n03376595": 18,
    "n03445777": 19,
    "n03452741": 20,
    "n03584829": 21,
    "n03590841": 22,
    "n03709823": 23,
    "n03773504": 24,
    "n03775071": 25,
    "n03792782": 26,
    "n03792972": 27,
    "n03877472": 28,
    "n03888257": 29,
    "n03982430": 30,
    "n04044716": 31,
    "n04069434": 32,
    "n04086273": 33,
    "n04120489": 34,
    "n04555897": 35,
    "n07753592": 36,
    "n07873807": 37,
    "n11939491": 38,
    "n13054560": 39,
}


def thread_read_write(x, y, pkl_filename):
    """Writes and dumps the processed pkl file for each stimulus(or called subject).
    [time, channels=127], y
    """
    with open(pkl_filename + ".pkl", "wb") as file:
        pickle.dump(x, file)
        pickle.dump(y, file)


def time_norm(x, dim=0, eps=1e-6):
    mean = np.mean(x, axis=dim, keepdims=True)
    std = np.std(x, axis=dim, keepdims=True)
    normalized_x = (x - mean) / (std + eps)
    return normalized_x


def get_patches_from_eeg(eeg, patch_size=32):
    # 400, 96, 512 -> 400, 512, 96
    eeg = eeg.transpose((0, 2, 1))
    locs_2d = np.loadtxt("locs_2d.csv", delimiter=",")
    x_min, x_max = locs_2d[:, 0].min(), locs_2d[:, 0].max()
    y_min, y_max = locs_2d[:, 1].min(), locs_2d[:, 1].max()

    grid_x, grid_y = np.mgrid[
        x_min : x_max : patch_size * 1j, y_min : y_max : patch_size * 1j
    ]
    eeg = eeg.reshape(-1, eeg.shape[-1])
    patches = []
    for i in tqdm(range(eeg.shape[0])):
        v_min = np.min(eeg[i])
        patch = griddata(
            locs_2d, eeg[i], (grid_x, grid_y), method="cubic", fill_value=v_min
        )

        patches.append(patch)

    patches = np.array(patches).reshape(400, 512, patch_size, patch_size)
    return patches


class LabelReader(object):
    def __init__(self, one_hot=False):
        self.file_path = None  # '/CVPR2021-02785/design/run-00.txt'
        self.one_hot = one_hot
        self.lines = None

    def read(self):
        with open(self.file_path) as f:
            lines = f.readlines()
        return [line.split("_")[0] for line in lines]

    def get_set(self, file_path):
        if self.file_path == file_path:
            return [classes[e] for e in self.lines]
        else:
            self.file_path = file_path
            self.lines = self.read()
            return [classes[e] for e in self.lines]


# read [b, c, t]
def read_auto(file_path):
    raw = mne.io.read_raw_bdf(
        file_path, preload=True, exclude=exclude, stim_channel=stim_channel
    )
    events = mne.find_events(
        raw,
        stim_channel="Status",
        output="step",
    )
    event_dict = {"stim": 65281, "end": 0}
    epochs = mne.Epochs(
        raw, events, event_id=event_dict, preload=True, tmin=-1e-3, tmax=2
    ).drop_channels("Status")
    epochs.equalize_event_counts(["stim"])
    stim_epochs = epochs["stim"]

    del raw, epochs, events
    return stim_epochs.get_data()[..., -8192:]


def file_scanf(path, endswith, sub_ratio=1):
    files = glob.glob(path + "/*")
    if platform.system().lower() == "windows":
        files = [f.replace("\\", "/") for f in files]
    disallowed_file_endings = (".gitignore", ".DS_Store")
    _input_files = files[: int(len(files) * sub_ratio)]
    return list(
        filter(
            lambda x: not x.endswith(disallowed_file_endings) and x.endswith(endswith),
            _input_files,
        )
    )


def get_filenames(dir_path):
    data_filenames = file_scanf(dir_path + "data", endswith=".bdf")
    label_filenames = file_scanf(dir_path + "design", endswith=".txt")
    data_filenames.sort()
    label_filenames.sort()
    return data_filenames, label_filenames


def go_through(data_filenames, label_filenames, pkl_path):

    for data_file, label_file in tqdm(
        zip(data_filenames, label_filenames),
        desc=" Total",
        position=0,
        leave=True,
        colour="YELLOW",
        ncols=80,
        # ncols=128,
    ):

        y = LabelReader().get_set(label_file)
        # (400, 96, 8192)
        eeg = read_auto(data_file)
        # 8192 -> 2048
        eeg = eeg[..., ::4]
        # first 0.5s
        eeg = eeg[:, :, :512]

        eeg = time_norm(eeg, dim=2)

        patch_size = 32
        imgs = get_patches_from_eeg(eeg, patch_size=patch_size)
        name = data_file.split("/")[-1].replace(".bdf", "")
        Parallel(n_jobs=parallel_jobs)(
            delayed(thread_read_write)(
                imgs[i],
                y[i],
                pkl_path
                + str(patch_size)
                + "x"
                + str(patch_size)
                + "/"
                + name
                + "_"
                + str(i)
                + "_"
                + str(y[i])
                + "_"
                + str(patch_size)
                + "x"
                + str(patch_size),
            )
            for i in tqdm(
                range(len(y)),
                desc=" write " + name,
                position=1,
                leave=False,
                colour="WHITE",
                # ncols=128,
                ncols=80,
            )
        )


if __name__ == "__main__":
    data_path = "/data1/share_data/CVPR2021-02785/"
    out_path = "/data1/share_data/purdue/patch/"
    data_filenames, label_filenames = get_filenames(data_path)
    go_through(data_filenames, label_filenames, pkl_path=out_path)
