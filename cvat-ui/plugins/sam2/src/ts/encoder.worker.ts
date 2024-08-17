import { InferenceSession, env, Tensor } from 'onnxruntime-web';

let encoder: InferenceSession | null = null;

env.wasm.wasmPaths = '/assets/';

export enum EncoderWorkerAction {
    INIT = 'init',
    ENCODE = 'encode',
}

export interface InitEncodeBody {
    encoderURL: string;
}

export interface EncodeBody {
    image_tensor: Tensor;  // The tensor of the input image [1, 3, 1024, 1024]
}

export interface WorkerOutput {
    action: EncoderWorkerAction;
    error?: string;
}

export interface WorkerInput {
    action: EncoderWorkerAction;
    payload: InitEncodeBody | EncodeBody;
}

const errorToMessage = (error: unknown): string => {
    if (error instanceof Error) {
        return error.message;
    }
    if (typeof error === 'string') {
        return error;
    }

    console.error(error);
    return 'Unknown error, please check console';
};

// eslint-disable-next-line no-restricted-globals
if ((self as any).importScripts) {
    onmessage = (e: MessageEvent<WorkerInput>) => {
        if (e.data.action === EncoderWorkerAction.INIT) {
            if (encoder) {
                return;
            }

            const body = e.data.payload as InitEncodeBody;
            InferenceSession.create(body.encoderURL).then((encoderSession) => {
                encoder = encoderSession;
                postMessage({ action: EncoderWorkerAction.INIT });
            }).catch((error: unknown) => {
                postMessage({ action: EncoderWorkerAction.INIT, error: errorToMessage(error) });
            });
        } else if (!encoder) {
            postMessage({
                action: e.data.action,
                error: 'Worker was not initialized',
            });
        } else if (e.data.action === EncoderWorkerAction.ENCODE) {
            const inputs = {
                image: (e.data.payload as EncodeBody).image_tensor,  // Pass the image tensor for encoding
            };

            encoder.run(inputs).then((results) => {
                postMessage({
                    action: EncoderWorkerAction.ENCODE,
                    payload: results,  // Embeddings or other encoder outputs
                });
            }).catch((error: unknown) => {
                postMessage({ action: EncoderWorkerAction.ENCODE, error: errorToMessage(error) });
            });
        }
    };
}
