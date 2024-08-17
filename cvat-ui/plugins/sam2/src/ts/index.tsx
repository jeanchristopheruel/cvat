import { Tensor } from 'onnxruntime-web';
import { CVATCore, MLModel, Job } from 'cvat-core-wrapper';
import { PluginEntryPoint, APIWrapperEnterOptions, ComponentBuilder } from 'components/plugins-entrypoint';
import { InitDecodeBody, DecodeBody, DecoderWorkerAction } from './decoder.worker';
import { InitEncodeBody, EncodeBody, EncoderWorkerAction } from './encoder.worker';
import { prepareDecodingInputs } from './decodingUtils'; // Assuming this file has prepareDecodingInputs function

/**
 * Helper to initialize workers
 */
function initializeWorker(worker: Worker, action: string, url: string, initBody: InitEncodeBody | InitDecodeBody, reject: (error: Error) => void) {
    worker.postMessage({ action, payload: initBody });
    return new Promise<void>((resolve, reject) => {
        worker.onmessage = (e: MessageEvent) => {
            if (e.data.error) {
                reject(new Error(`${action} worker initialization failed: ${e.data.error}`));
            } else {
                resolve();
            }
        };
    });
}

/**
 * Helper to convert ImageData into a tensor
 */
async function convertImageToTensor(imageData: ImageData): Promise<Tensor> {
    const { width, height, data } = imageData;
    const normalizedData = new Float32Array(width * height * 3);
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    for (let i = 0, index = 0; i < data.length; i += 4, index += 3) {
        normalizedData[index] = (data[i] / 255 - mean[0]) / std[0];     // Red
        normalizedData[index + 1] = (data[i + 1] / 255 - mean[1]) / std[1]; // Green
        normalizedData[index + 2] = (data[i + 2] / 255 - mean[2]) / std[2]; // Blue
    }

    return new Tensor('float32', normalizedData, [1, 3, height, width]);
}

/**
 * Fetch ImageBitmap and convert it into a tensor
 */
async function fetchImageTensor(job: Job, frame: number): Promise<Tensor> {
    const imageBitmaps = await job.frames.contextImage(frame);
    const imageBitmap = imageBitmaps['main'];

    if (!imageBitmap) {
        throw new Error('Failed to retrieve ImageBitmap for frame.');
    }

    const canvas = document.createElement('canvas');
    canvas.width = imageBitmap.width;
    canvas.height = imageBitmap.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Failed to create canvas context.');

    ctx.drawImage(imageBitmap, 0, 0);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return convertImageToTensor(imageData);
}

interface ClickType {
    clickType: 0 | 1;
    x: number;
    y: number;
}


/**
 * Main plugin object
 */
const sam2Plugin: SAM2Plugin = {
    name: 'Segment Anything',
    description: 'Handles non-default SAM serverless function output',
    cvat: {
        jobs: {
            get: {
                async leave(plugin: SAM2Plugin, results: any[], query: { jobID?: number }): Promise<any> {
                    if (typeof query.jobID === 'number') {
                        [plugin.data.jobs[query.jobID]] = results;
                    }
                    return results;
                },
            },
        },
        lambda: {
            call: {
                async enter(plugin: SAM2Plugin, taskID: number, model: MLModel, { frame }: { frame: number }): Promise<null | APIWrapperEnterOptions> {
                    return new Promise(async (resolve, reject) => {
                        try {
                            if (!plugin.data.initialized) {
                                await initializeWorker(sam2Plugin.data.encoderWorker, EncoderWorkerAction.INIT, sam2Plugin.data.encoderURL, { encoderURL: sam2Plugin.data.encoderURL }, reject);
                                await initializeWorker(sam2Plugin.data.decoderWorker, DecoderWorkerAction.INIT, sam2Plugin.data.decoderURL, { decoderURL: sam2Plugin.data.decoderURL }, reject);
                                sam2Plugin.data.initialized = true;
                            }

                            const imageTensor = await fetchImageTensor(plugin.data.jobs[taskID], frame);
                            plugin.data.encoderWorker.postMessage({
                                action: EncoderWorkerAction.ENCODE,
                                payload: { image_tensor: imageTensor },
                            });

                            plugin.data.encoderWorker.onmessage = (e: MessageEvent) => {
                                if (e.data.error) reject(e.data.error);
                                else resolve(null);
                            };
                        } catch (error) {
                            console.error(`Error processing frame ${frame}:`, error);
                            reject(error);
                        }
                    });
                },

                async leave(plugin: SAM2Plugin, result: any, taskID: number, model: MLModel, { frame, pos_points, neg_points }: { frame: number, pos_points: number[][], neg_points: number[][] }): Promise<{ mask: number[][]; bounds: [number, number, number, number] }> {
                    return new Promise(async (resolve, reject) => {
                        try {
                            const composedClicks = [...pos_points, ...neg_points].map(([x, y], index) => ({
                                clickType: index < pos_points.length ? 1 : 0,
                                x, y,
                            }));

                            const pointCoords = new Tensor('float32', new Float32Array(composedClicks.flatMap(click => [click.x, click.y])), [1, composedClicks.length, 2]);
                            const pointLabels = new Tensor('float32', new Float32Array(composedClicks.map(click => click.clickType)), [1, composedClicks.length]);

                            const decodeInputs = prepareDecodingInputs(plugin.data.encoderResults, pointCoords, pointLabels);

                            plugin.data.decoderWorker.postMessage({
                                action: DecoderWorkerAction.DECODE,
                                payload: decodeInputs,
                            });

                            plugin.data.decoderWorker.onmessage = (e: MessageEvent) => {
                                if (e.data.error) reject(e.data.error);
                                else resolve({ mask: e.data.payload.masks, bounds: [e.data.payload.xtl, e.data.payload.ytl, e.data.payload.xbr, e.data.payload.ybr] });
                            };
                        } catch (error) {
                            reject(error);
                        }
                    });
                },
            },
        },
    },
    data: {
        initialized: false,
        core: null,
        encoderWorker: new Worker(new URL('./encoder.worker', import.meta.url)),
        decoderWorker: new Worker(new URL('./decoder.worker', import.meta.url)),
        jobs: {},
        encoderURL: '/assets/sam2_hiera_large_encoder.with_runtime_opt.ort',
        decoderURL: '/assets/decoder.onnx',
        encoderResults: null,
    },
    callbacks: {
        onStatusChange: null,
    },
};

/**
 * Register the plugin with the CVAT UI
 */
const builder: ComponentBuilder = ({ core }) => {
    sam2Plugin.data.core = core;
    core.plugins.register(sam2Plugin);

    return { name: sam2Plugin.name, destructor: () => {} };
};

function register(): void {
    if (Object.prototype.hasOwnProperty.call(window, 'cvatUI')) {
        (window as any as { cvatUI: { registerComponent: PluginEntryPoint } }).cvatUI.registerComponent(builder);
    }
}

window.addEventListener('plugins.ready', register, { once: true });