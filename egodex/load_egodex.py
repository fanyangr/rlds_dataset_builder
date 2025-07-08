"""
Script to load and work with the EgoDex dataset.

The EgoDex dataset consists of 800+ hours of 30 Hz, 1080p egocentric video and paired 
30 Hz 3D pose annotations for the upper body, hands, and camera extrinsics. 
The data consists entirely of tabletop manipulation tasks across ~200 diverse tasks.

Based on the official EgoDex README format specification.

"""

import os
import h5py
import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
# Default camera intrinsics for EgoDex dataset (from README)
DEFAULT_INTRINSIC = np.array([
    [736.6339, 0., 960.], 
    [0., 736.6339, 540.], 
    [0., 0., 1.]
])

# JOINT_NAMES_OF_INTEREST = ['leftHand', 'leftThumbTip', 'leftIndexFingerTip', 'leftMiddleFingerTip', 'leftRingFingerTip', 'leftLittleFingerTip',
#                            'rightHand', 'rightThumbTip', 'rightIndexFingerTip', 'rightMiddleFingerTip', 'rightRingFingerTip', 'rightLittleFingerTip',
#                            'camera']
JOINT_NAMES_OF_INTEREST = ['leftHand', 'leftThumbTip', 'leftIndexFingerTip', 'leftMiddleFingerTip',
                           'rightHand', 'rightThumbTip', 'rightIndexFingerTip', 'rightMiddleFingerTip',
                           'camera']
HAND_JOINT_NAMES_OF_INTEREST = ['leftHand', 'rightHand', 'camera']
# Dataset specifications from README
DATASET_FPS = 30  # 30 Hz video and pose data
VIDEO_RESOLUTION = (1080, 1920)  # 1080p video (height, width)


class EgoDexEpisode:
    """
    Represents a single episode from the EgoDex dataset.
    
    Each episode consists of a paired HDF5 file (pose annotations) and MP4 file (video).
    The HDF5 file contains:
    - camera/intrinsic: 3x3 camera intrinsics  
    - transforms/[joint_name]: N x 4 x 4 SE(3) transforms for each joint
    - confidences/[joint_name]: N scalar confidence values (0-1)
    - Language annotations in file attributes
    """
    
    def __init__(self, hdf5_path: str, video_path: str, image_size: Optional[Tuple[int, int]]=(224, 224)):
        self.hdf5_path = hdf5_path
        self.video_path = video_path
        self.task_name = Path(hdf5_path).parent.name
        self.episode_id = Path(hdf5_path).stem
        self.image_size = image_size
        self.include_rotations = False
        self._load_data()
    
    def _load_data(self):
        """Load data from HDF5 file according to README specification."""
        with h5py.File(self.hdf5_path, 'r') as f:
            # Load camera intrinsics (3x3 matrix)
            self.intrinsic = DEFAULT_INTRINSIC
            
            # Load all joint transforms (N x 4 x 4 SE(3) poses)
            self.transforms = {}
            self.joint_names = []
            if 'transforms' in f:
                for joint_name in f['transforms'].keys():
                    self.transforms[joint_name] = f[f'transforms/{joint_name}'][()]
                    self.joint_names.append(joint_name)
            # Load confidences (N scalar values per joint)
            self.confidences = {}
            if 'confidences' in f:
                for joint_name in f['confidences'].keys():
                    self.confidences[joint_name] = f[f'confidences/{joint_name}'][()]
            
            # Load language description from file attributes
            self.language_description = f.attrs.get('llm_description', 'No description available')
            if isinstance(self.language_description, bytes):
                self.language_description = self.language_description.decode('utf-8')
            
            # For reversible tasks, there might be a second description
            self.language_description2 = f.attrs.get('llm_description2', None)
            if self.language_description2 and isinstance(self.language_description2, bytes):
                self.language_description2 = self.language_description2.decode('utf-8')
        
        # Get video properties - should be 30 FPS according to README
        cap = cv2.VideoCapture(self.video_path)
        self.frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.duration = self.frame_count / self.fps if self.fps > 0 else 0
        cap.release()
        
        # Verify N = 30 * T relationship from README
        expected_poses = int(self.duration * DATASET_FPS)
        if len(self.transforms) > 0:
            actual_poses = len(next(iter(self.transforms.values())))
            if abs(expected_poses - actual_poses) > 1:  # Allow 1 frame tolerance
                print(f"Warning: Expected {expected_poses} poses, got {actual_poses} in {self.hdf5_path}")
    
    def get_frame(self, frame_idx: int) -> np.ndarray:
        """Get a specific video frame (1080p according to README)."""
        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Could not read frame {frame_idx} from {self.video_path}")
        return frame
    
    def get_transforms_at_frame(self, frame_idx: int, joint_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Get SE(3) transforms (4x4 matrices) for specified joints at a given frame.
        All transforms are in the ARKit origin frame (stationary ground frame).
        """
        if joint_names is None:
            joint_names = self.joint_names
        
        transforms = {}
        for joint_name in joint_names:
            if joint_name in self.transforms:
                if frame_idx < len(self.transforms[joint_name]):
                    transforms[joint_name] = self.transforms[joint_name][frame_idx]
        return transforms
    
    def get_positions_at_frame(self, frame_idx: int, joint_names: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """Get 3D positions (translation part of SE(3)) for specified joints at a given frame."""
        transforms = self.get_transforms_at_frame(frame_idx, joint_names)
        positions = {}
        for joint_name, transform in transforms.items():
            positions[joint_name] = transform[:3, 3]  # Extract translation vector
        return positions
    
    def get_confidences_at_frame(self, frame_idx: int, joint_names: Optional[List[str]] = None) -> Dict[str, float]:
        """Get confidence values (0-1) for specified joints at a given frame."""
        if joint_names is None:
            joint_names = self.joint_names
        
        confidences = {}
        for joint_name in joint_names:
            if joint_name in self.confidences:
                if frame_idx < len(self.confidences[joint_name]):
                    confidences[joint_name] = self.confidences[joint_name][frame_idx]
        return confidences
    
    def get_state_vector(self, frame_idx: int, joint_names: Optional[List[str]] = None, 
                        include_camera: bool = True, include_rotations: bool = False) -> np.ndarray:
        """
        Get flattened state vector for specified joints at a given frame.
        
        Args:
            frame_idx: Frame index
            joint_names: List of joints to include
            include_rotations: If True, include rotation matrices (9 values per joint)
                             If False, only include positions (3 values per joint)
                             Note: Camera always includes both position and rotation regardless of this flag
        """
        if joint_names is None:
            joint_names = self.joint_names
        
        features = []
        for joint_name in joint_names:
            if joint_name in self.transforms and frame_idx < len(self.transforms[joint_name]):
                transform = self.transforms[joint_name][frame_idx]
                # Always include position
                position = transform[:3, 3]
                
                # Include rotation for camera always, or for other joints if requested
                if (joint_name == 'camera' and include_camera) or (joint_name != 'camera' and include_rotations):
                    # if include camera, always include rotation
                    # rotation = transform[:3, :3].flatten()
                    # translation = transform[:3, 3]
                    # camera_pose = np.concatenate([translation, rotation])
                    # features.extend(camera_pose)
                    # Convert rotation matrix to quaternion using scipy
                    rotation_matrix = transform[:3, :3]
                    quaternion = R.from_matrix(rotation_matrix).as_quat().astype(np.float32)  # Returns [x, y, z, w]
                    camera_pose = np.concatenate([position, quaternion])
                    features.extend(camera_pose)
                elif joint_name == "camera" and not include_camera:
                    continue
                else:
                    features.extend(position)
            else:
                print("data missing")
                # If joint not available, pad with zeros
                # Camera gets 12 values (3 pos + 9 rot), others get 3 or 12 based on include_rotations
                if joint_name == 'camera':
                    pad_size = 12 if include_camera else 0
                else:
                    pad_size = 12 if include_rotations else 3
                features.extend([0.0] * pad_size)
        
        return np.array(features, dtype=np.float32)
    
    def get_camera_extrinsics(self, frame_idx: int) -> np.ndarray:
        """Get camera extrinsics (camera transform in ARKit origin frame)."""
        if 'camera' in self.transforms and frame_idx < len(self.transforms['camera']):
            return self.transforms['camera'][frame_idx]
        else:
            raise ValueError(f"Camera extrinsics not available for frame {frame_idx}")
    
    def __len__(self) -> int:
        """Return number of frames in the episode."""
        return self.frame_count
    
    def __repr__(self) -> str:
        return f"EgoDexEpisode(task='{self.task_name}', id='{self.episode_id}', frames={self.frame_count}, duration={self.duration:.2f}s)"
    
    def get_trajectory(self, joint_names: Optional[List[str]] = None, 
                      start_frame: int = 0, end_frame: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Get 3D trajectory for specified joints over a frame range.
        
        Args:
            joint_names: List of joints to include
            start_frame: Starting frame index
            end_frame: Ending frame index (if None, uses all frames)
            
        Returns:
            Dictionary mapping joint names to (N, 3) position arrays
        """
        if joint_names is None:
            joint_names = self.joint_names
        
        if end_frame is None:
            end_frame = self.frame_count
        
        trajectories = {}
        for joint_name in joint_names:
            if joint_name in self.transforms:
                positions = []
                for frame_idx in range(start_frame, min(end_frame, len(self.transforms[joint_name]))):
                    transform = self.transforms[joint_name][frame_idx]
                    position = transform[:3, 3]
                    positions.append(position)
                trajectories[joint_name] = np.array(positions)
        
        return trajectories
    def __getitem__(self, frame_idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        """Get a single data sample."""
                # Get video frame
        frame = self.get_frame(frame_idx)
        if self.image_size is not None:
            h, w = frame.shape[:2]
            center_x = w // 2
            center_y = h // 2
            crop_size = min(h, w)  # This will be 1080 since height is 1080
            start_x = center_x - crop_size // 2
            start_y = center_y - crop_size // 2
            frame = frame[start_y:start_y + crop_size, start_x:start_x + crop_size]
            
            # Now resize the square frame to target size
            frame = cv2.resize(frame, (self.image_size[1], self.image_size[0]))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        
        # Get state vector (3D positions and optionally rotations)
        state = self.get_state_vector(frame_idx=frame_idx, joint_names=JOINT_NAMES_OF_INTEREST, 
                                         include_rotations=self.include_rotations, include_camera=True)
        # state = torch.from_numpy(state)
        
        # Get confidence values
        confidences = self.get_confidences_at_frame(frame_idx, self.joint_names)
        confidence_values = np.array([confidences.get(joint, 0.0) for joint in self.joint_names], dtype=np.float32)
        
        # Get camera intrinsics
        # intrinsics = torch.from_numpy(self.intrinsic.astype(np.float32))
        
        return {
            'image': frame,
            'state': state,
            'confidences': confidence_values,
            'intrinsics': self.intrinsic,
            'frame_idx': frame_idx,
            'task_name': self.task_name,
            'episode_id': self.episode_id,
            'language_description': self.language_description,
            'language_description2': self.language_description2 or "",
            'camera': self.get_camera_extrinsics(frame_idx).flatten().astype(np.float32),
        }
    


class EgoDexDataset(Dataset):
    """
    PyTorch Dataset for loading EgoDex data.
    
    Supports loading from different parts of the dataset:
    - test/ directory (1% of data)
    - part1/, part2/, etc. directories (training data)
    - extra/ directory (additional samples)
    """
    
    def __init__(self, data_dir: str, tasks: Optional[List[str]] = None, 
                 joint_names: Optional[List[str]] = None, 
                 image_size: Optional[Tuple[int, int]] = None,
                 include_rotations: bool = False,
                 confidence_threshold: float = 0.0):
        """
        Initialize EgoDex dataset.
        
        Args:
            data_dir: Path to the dataset directory (e.g., 'test/', 'part1/', etc.)
            tasks: List of task names to include (if None, includes all tasks)
            joint_names: List of joint names to include in state vector (if None, uses all)
            image_size: Target size for images (height, width). If None, keeps original 1080p
            include_rotations: Whether to include rotation matrices in state vector
            confidence_threshold: Minimum confidence threshold for including frames
        """
        self.data_dir = Path(data_dir)
        self.tasks = tasks
        self.joint_names = joint_names
        self.image_size = image_size
        self.include_rotations = include_rotations
        self.confidence_threshold = confidence_threshold
        
        self.episodes = self._discover_episodes()
        self.frame_indices = self._build_frame_index()
        
        # Determine joint names to use
        if self.joint_names is None and len(self.episodes) > 0:
            self.joint_names = self.episodes[0].joint_names
    
    def _discover_episodes(self) -> List[EgoDexEpisode]:
        """Discover all episodes in the dataset according to README structure."""
        episodes = []
        
        # First pass: count total files for progress bar
        total_files = 0
        task_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        for task_dir in task_dirs:
            task_name = task_dir.name
            if self.tasks is not None and task_name not in self.tasks:
                continue
            total_files += len(list(task_dir.glob("*.hdf5")))
        
        # Second pass: process files with progress bar
        with tqdm(total=total_files, desc="Discovering episodes", unit="files") as pbar:
            for task_dir in task_dirs:
                if not task_dir.is_dir():
                    continue
                
                task_name = task_dir.name
                if self.tasks is not None and task_name not in self.tasks:
                    continue
                
                # Find all HDF5 files in the task directory
                for hdf5_file in sorted(task_dir.glob("*.hdf5")):
                    # Look for corresponding MP4 file with same index
                    video_file = hdf5_file.with_suffix(".mp4")
                    if video_file.exists():
                        try:
                            episode = EgoDexEpisode(str(hdf5_file), str(video_file))
                            episodes.append(episode)
                        except Exception as e:
                            print(f"Warning: Could not load episode {hdf5_file}: {e}")
                    else:
                        print(f"Warning: No corresponding video file for {hdf5_file}")
                    pbar.update(1)
        
        print(f"Discovered {len(episodes)} episodes across {len(set(ep.task_name for ep in episodes))} tasks")
        return episodes
    
    def _build_frame_index(self) -> List[Tuple[int, int]]:
        """Build index of (episode_idx, frame_idx) for all valid frames."""
        frame_indices = []
        
        # Calculate total frames for progress bar
        total_frames = sum(len(episode) for episode in self.episodes)
        
        with tqdm(total=total_frames, desc="Building frame index", unit="frames") as pbar:
            for ep_idx, episode in enumerate(self.episodes):
                for frame_idx in range(len(episode)):
                    # Apply confidence filtering if specified
                    if self.confidence_threshold > 0.0:
                        confidences = episode.get_confidences_at_frame(frame_idx, self.joint_names)
                        if confidences and min(confidences.values()) < self.confidence_threshold:
                            pbar.update(1)
                            continue
                    
                    frame_indices.append((ep_idx, frame_idx))
                    pbar.update(1)
        
        return frame_indices
    
    def __len__(self) -> int:
        return len(self.frame_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        """Get a single data sample."""
        ep_idx, frame_idx = self.frame_indices[idx]
        episode = self.episodes[ep_idx]
        
        # Get video frame
        frame = episode.get_frame(frame_idx)
        if self.image_size is not None:
            h, w = frame.shape[:2]
            center_x = w // 2
            center_y = h // 2
            crop_size = min(h, w)  # This will be 1080 since height is 1080
            start_x = center_x - crop_size // 2
            start_y = center_y - crop_size // 2
            frame = frame[start_y:start_y + crop_size, start_x:start_x + crop_size]
            
            # Now resize the square frame to target size
            frame = cv2.resize(frame, (self.image_size[1], self.image_size[0]))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        
        # Get state vector (3D positions and optionally rotations)
        state = episode.get_state_vector(frame_idx=frame_idx, joint_names=self.joint_names, 
                                         include_rotations=self.include_rotations, include_camera=True)
        state = torch.from_numpy(state)
        
        # Get confidence values
        confidences = episode.get_confidences_at_frame(frame_idx, self.joint_names)
        confidence_values = np.array([confidences.get(joint, 0.0) for joint in self.joint_names], dtype=np.float32)
        
        # Get camera intrinsics
        intrinsics = torch.from_numpy(episode.intrinsic.astype(np.float32))
        
        return {
            'image': frame,
            'state': state,
            'confidences': torch.from_numpy(confidence_values),
            'intrinsics': intrinsics,
            'episode_idx': ep_idx,
            'frame_idx': frame_idx,
            'task_name': episode.task_name,
            'episode_id': episode.episode_id,
            'language_description': episode.language_description,
            'language_description2': episode.language_description2 or "",
            'camera': episode.get_camera_extrinsics(frame_idx).flatten().astype(np.float32),
        }
    
    def get_episode(self, idx: int) -> EgoDexEpisode:
        """Get a specific episode."""
        return self.episodes[idx]
    
    def get_task_names(self) -> List[str]:
        """Get list of all task names in the dataset."""
        return sorted(list(set(episode.task_name for episode in self.episodes)))
    
    def get_joint_names(self) -> List[str]:
        """Get list of all available joint names."""
        if len(self.episodes) > 0:
            return self.episodes[0].joint_names
        return []
    
    def get_dataset_stats(self) -> Dict[str, Union[int, float, List[str]]]:
        """Get comprehensive dataset statistics."""
        if len(self.episodes) == 0:
            return {}
        
        total_frames = sum(len(ep) for ep in self.episodes)
        total_duration = sum(ep.duration for ep in self.episodes)
        task_counts = {}
        for ep in self.episodes:
            task_counts[ep.task_name] = task_counts.get(ep.task_name, 0) + 1
        
        return {
            'num_episodes': len(self.episodes),
            'num_tasks': len(set(ep.task_name for ep in self.episodes)),
            'total_frames': total_frames,
            'total_duration_hours': total_duration / 3600,
            'avg_episode_duration': total_duration / len(self.episodes),
            'fps': DATASET_FPS,
            'task_distribution': task_counts,
            'joint_names': self.get_joint_names(),
            'num_joints': len(self.get_joint_names()),
            'state_vector_size': len(self.get_joint_names()) * (3 + (9 if self.include_rotations else 0))
        }


def print_dataset_info(dataset: EgoDexDataset):
    """Print comprehensive information about the dataset."""
    stats = dataset.get_dataset_stats()
    
    print("=" * 60)
    print("EgoDex Dataset Information")
    print("=" * 60)
    print(f"Dataset directory: {dataset.data_dir}")
    print(f"Total episodes: {stats.get('num_episodes', 0)}")
    print(f"Total tasks: {stats.get('num_tasks', 0)}")
    print(f"Total frames: {stats.get('total_frames', 0):,}")
    print(f"Total duration: {stats.get('total_duration_hours', 0):.2f} hours")
    print(f"Average episode duration: {stats.get('avg_episode_duration', 0):.2f} seconds")
    print(f"Dataset FPS: {stats.get('fps', 0)} Hz")
    print(f"Number of joints: {stats.get('num_joints', 0)}")
    print(f"State vector size: {stats.get('state_vector_size', 0)}")
    
    print(f"\nTask distribution:")
    task_dist = stats.get('task_distribution', {})
    for task, count in sorted(task_dist.items()):
        print(f"  {task}: {count} episodes")
    
    print(f"\nAvailable joints:")
    joint_names = stats.get('joint_names', [])
    for i, joint in enumerate(joint_names):
        if i % 4 == 0:
            print()
        print(f"  {joint:<20}", end="")
    print()
    
    if len(dataset.episodes) > 0:
        print(f"\nExample episode:")
        example_ep = dataset.episodes[0]
        print(f"  {example_ep}")
        print(f"  Language: {example_ep.language_description[:100]}...")
        
        # Check if confidences are available
        sample_confidences = example_ep.get_confidences_at_frame(0)
        print(f"  Confidences available: {'Yes' if sample_confidences else 'No'}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Load and explore EgoDex dataset")
    parser.add_argument("--data_dir", type=str, default="/root/raw_datasets/egodex/small_test/",
                        help="Path to the EgoDex dataset directory (test/, part1/, etc.)")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Specific tasks to load (default: all tasks)")
    parser.add_argument("--joints", nargs="*", default=None,
                        help="Specific joints to include (default: all joints)")
    parser.add_argument("--image_size", nargs=2, type=int, default=None,
                        help="Target image size (height width). Default: keep 1080p")
    parser.add_argument("--include_rotations", action="store_true",
                        help="Include rotation matrices in state vector")
    parser.add_argument("--confidence_threshold", type=float, default=0.0,
                        help="Minimum confidence threshold for including frames")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for data loader")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of worker processes for data loading")
    
    args = parser.parse_args()
    
    # Expand path
    data_dir = os.path.expanduser(args.data_dir)
    
    # Create dataset
    print(f"Loading EgoDex dataset from: {data_dir}")
    dataset = EgoDexDataset(
        data_dir=data_dir, 
        tasks=args.tasks,
        joint_names=args.joints,
        image_size=tuple(args.image_size) if args.image_size else None,
        include_rotations=args.include_rotations,
        confidence_threshold=args.confidence_threshold
    )
    
    # Print dataset information
    print_dataset_info(dataset)
    
    if len(dataset) == 0:
        print("No data found! Please check the data directory path.")
        return

    # Load an episode
    episode = dataset.get_episode(0)

    # Create interactive 3D trajectory visualization with keypoints of interest
    # Use JOINT_NAMES_OF_INTEREST to visualize the predefined important joints
    fig = episode.plot_3d_trajectories_plotly(
        joint_names=JOINT_NAMES_OF_INTEREST,  # Changed: Use the predefined joints of interest
        start_frame=0,                        # Optional: start frame
        # end_frame=100,                        # Optional: end frame  
        save_path='keypoints_of_interest_trajectory.html'  # Updated filename
    )
    
    # Print which joints are being visualized
    print(f"\nVisualizing trajectories for joints of interest: {JOINT_NAMES_OF_INTEREST}")

if __name__ == "__main__":
    main()