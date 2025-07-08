from typing import Iterator, Tuple, Any
import os
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
from egodex.load_egodex import EgoDexEpisode, JOINT_NAMES_OF_INTEREST
import glob

TARGET_IMAGE_HEIGHT = 224
TARGET_IMAGE_WIDTH = 224

class EgoDex(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        # 'wrist_image': tfds.features.Image(
                        #     shape=(64, 64, 3),
                        #     dtype=np.uint8,
                        #     encoding_format='png',
                        #     doc='Wrist camera RGB observation.',
                        # ),                        
                        'state': tfds.features.Tensor(
                            shape=(31,),
                            dtype=np.float32,
                            doc='Hand state, consists of [18x hand keypoints coordinates, '
                                '7x camera pose].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(24,),
                        dtype=np.float32,
                        doc='Hand action, consists of [18x hand keypoints coordinates].',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/home/ANT.AMAZON.COM/fanyangr/Downloads/small_test'),
            # 'val': self._generate_examples(path='data/val/episode_*.npy'),
        }

    def _discover_episodes(self, data_dir: str):
        """Discover all episodes in the dataset directory."""
        episodes = []
        data_path = Path(data_dir)
        
        # Iterate through task directories
        for task_dir in data_path.iterdir():
            if not task_dir.is_dir():
                continue
                
            # Find all HDF5 files in the task directory
            for hdf5_file in sorted(task_dir.glob("*.hdf5")):
                # Look for corresponding MP4 file with same base name
                video_file = hdf5_file.with_suffix(".mp4")
                if video_file.exists():
                    try:
                        episode = EgoDexEpisode(str(hdf5_file), str(video_file), image_size=(TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH))
                        episodes.append(episode)
                    except Exception as e:
                        print(f"Warning: Could not load episode {hdf5_file}: {e}")
                else:
                    print(f"Warning: No corresponding video file for {hdf5_file}")
        
        return episodes

    def _process_image(self, frame: np.ndarray) -> np.ndarray:
        """Process video frame to target size."""
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Center crop to square
        h, w = frame.shape[:2]
        center_x = w // 2
        center_y = h // 2
        crop_size = min(h, w)
        start_x = center_x - crop_size // 2
        start_y = center_y - crop_size // 2
        frame = frame[start_y:start_y + crop_size, start_x:start_x + crop_size]
        
        # Resize to target size
        frame = cv2.resize(frame, (TARGET_IMAGE_WIDTH, TARGET_IMAGE_HEIGHT))
        
        return frame.astype(np.uint8)

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        # def _parse_example(episode_path):
        #     data = EgoDexDataset(
        #         data_dir=path,
        #         tasks=None,
        #         joint_names=JOINT_NAMES_OF_INTEREST,
        #         image_size=(TARGET_IMAGE_HEIGHT, TARGET_IMAGE_WIDTH),
        #         include_rotations=False  # Use positions only for now
        #         )
        #     # load raw data --> this should change for your dataset
        #     # data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case

        #     # assemble episode --> here we're assuming demos so we set reward to 1 at the end
        #     episode = []
        #     for i, step in enumerate(data):
        #         # compute Kona language embedding
        #         language_embedding = self._embed([step['language_description']])[0].numpy()

        #         episode.append({
        #             'observation': {
        #                 'image': step['image'],
        #                 # 'wrist_image': step['wrist_image'],
        #                 'state': step['state'],
        #             },
        #             'action': step['state'][:-7],  # TODO: fix this later
        #             'discount': 1.0,
        #             'reward': float(i == (len(data) - 1)),
        #             'is_first': i == 0,
        #             'is_last': i == (len(data) - 1),
        #             'is_terminal': i == (len(data) - 1),
        #             'language_instruction': step['language_description'],
        #             'language_embedding': language_embedding,
        #         })

        #     # create output data sample
        #     sample = {
        #         'steps': episode,
        #         'episode_metadata': {
        #             'file_path': episode_path
        #         }
        #     }

        #     # if you want to skip an example for whatever reason, simply return None
        #     return episode_path, sample

        # # create list of all examples
        # episode_paths = glob.glob(path)

        # # for smallish datasets, use single-thread parsing
        # # for sample in episode_paths:
        # #     yield _parse_example(sample)

        # # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )






        # Discover all episodes
        episodes = self._discover_episodes(path)
        print(f"Found {len(episodes)} episodes")
        
        for episode in episodes:
            # Create episode data
            episode_steps = []
            episode_id = f"{episode.task_name}_{episode.episode_id}"
            
            for frame_idx in range(len(episode)-1):
                # Get video frame and process it
                frame_data = episode[frame_idx]
                
                # Compute language embedding
                language_embedding = self._embed([episode.language_description])[0].numpy()
                
                episode_steps.append({
                    'observation': {
                        'image': frame_data['image'],
                        'state': frame_data['state'],
                    },
                    'action': episode[frame_idx+1]['state'][:-7],
                    'discount': 1.0,
                    'reward': float(frame_idx == (len(episode) - 1)),  # Reward only at final step
                    'is_first': frame_idx == 0,
                    'is_last': frame_idx == (len(episode) - 1),
                    'is_terminal': frame_idx == (len(episode) - 1),
                    'language_instruction': episode.language_description,
                    'language_embedding': language_embedding.astype(np.float32),
                })
                    
            
            if episode_steps:  # Only yield if we have valid steps
                sample = {
                    'steps': episode_steps,
                    'episode_metadata': {
                        'file_path': episode.hdf5_path
                    }
                }
                yield episode_id, sample
