from .save_load_pose import TSSavePoseDataAsPickle, TSLoadPoseDataPickle
from .openpose_smoother import KPSSmoothPoseDataAndRender
from .load_video_batch import LoadVideoBatchListFromDir

NODE_CLASS_MAPPINGS = {
    "TSSavePoseDataAsPickle": TSSavePoseDataAsPickle,
    "TSLoadPoseDataPickle": TSLoadPoseDataPickle,
    "TSPoseDataSmoother": KPSSmoothPoseDataAndRender,
    "TSLoadVideoBatchListFromDir": LoadVideoBatchListFromDir,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TSSavePoseDataAsPickle": "TS Save Pose Data (PKL)",
    "TSLoadPoseDataPickle": "TS Load Pose Data (PKL)",
    "TSPoseDataSmoother": "TS Pose Data Smoother",
    "TSLoadVideoBatchListFromDir": "TS Load Video Batch List From Dir",
}
