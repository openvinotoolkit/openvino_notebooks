import './NotebooksList.scss';

import { INotebookMetadata } from '@/models/notebook';

import { NotebookCard } from './NotebookCard/NotebookCard';

const notebooks: INotebookMetadata[] = [
  {
    title: 'Hello Image Classification',
    description: 'This basic introduction to OpenVINO™ shows how to do inference with an image classification model.',
    additionalResources: null,
    imageUrl: 'https://user-images.githubusercontent.com/36741649/127172572-1cdab941-df5f-42e2-a367-2b334a3db6d8.jpg',
    links: {
      github:
        'https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/001-hello-world/001-hello-world.ipynb',
      colab:
        'https://colab.research.google.com/github/openvinotoolkit/openvino_notebooks/blob/main/notebooks/001-hello-world/001-hello-world.ipynb',
      binder:
        'https://mybinder.org/v2/gh/openvinotoolkit/openvino_notebooks/HEAD?filepath=notebooks%2F001-hello-world%2F001-hello-world.ipynb',
    },
    tags: {
      categories: ['getting-started'],
      tasks: ['cv', 'image-classification'],
      models: ['MobileNet'],
      libraries: [],
      other: [],
    },
  },
  {
    title: 'ControlNet - Stable-Diffusion',
    description: 'A Text-to-Image Generation with ControlNet Conditioning and OpenVINO™.',
    additionalResources: null,
    imageUrl: 'https://user-images.githubusercontent.com/29454499/224541412-9d13443e-0e42-43f2-8210-aa31820c5b44.png',
    links: {
      github:
        'https://github.com/openvinotoolkit/openvino_notebooks/blob/main/notebooks/235-controlnet-stable-diffusion/235-controlnet-stable-diffusion.ipynb',
    },
    tags: {
      categories: ['AI Trends'],
      tasks: ['Multimodal', 'Text-to-Image'],
      models: ['Stable Diffusion', 'ControlNet'],
      libraries: [],
      other: [],
    },
  },
];

export const NotebooksList = (): JSX.Element => {
  return (
    <div className="notebooks-container">
      {notebooks.map((notebook, i) => (
        <NotebookCard key={`notebook-${i}`} item={notebook}></NotebookCard>
      ))}
    </div>
  );
};
