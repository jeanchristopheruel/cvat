import { InferenceSession, env, Tensor } from 'onnxruntime-web';

let decoder: InferenceSession | null = null;

env.wasm.wasmPaths = '/assets/';

export enum DecoderWorkerAction {
    INIT = 'init',
    DECODE = 'decode',
}

export interface InitDecodeBody {
    decoderURL: string;
}

export interface DecodeBody {
    image_embed: Tensor;
    high_res_feats_0: Tensor;
    high_res_feats_1: Tensor;
    point_coords: Tensor;
    point_labels: Tensor;
    mask_input: Tensor;
    has_mask_input: Tensor;
    readonly [name: string]: Tensor;
}

export interface WorkerOutput {
    action: DecoderWorkerAction;
    error?: string;
}

export interface WorkerInput {
    action: DecoderWorkerAction;
    payload: InitDecodeBody | DecodeBody;
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
        if (e.data.action === DecoderWorkerAction.INIT) {
            if (decoder) {
                return;
            }

            const body = e.data.payload as InitDecodeBody;
            InferenceSession.create(body.decoderURL).then((decoderSession) => {
                decoder = decoderSession;
                postMessage({ action: DecoderWorkerAction.INIT });
            }).catch((error: unknown) => {
                postMessage({ action: DecoderWorkerAction.INIT, error: errorToMessage(error) });
            });
        } else if (!decoder) {
            postMessage({
                action: e.data.action,
                error: 'Worker was not initialized',
            });
        } else if (e.data.action === DecoderWorkerAction.DECODE) {
            const inputs = e.data.payload as DecodeBody;
            decoder.run(inputs).then((results) => {
                // Adjust the response based on the decoder output. Make sure to match the expected output structure.
                postMessage({
                    action: DecoderWorkerAction.DECODE,
                    payload: {
                        masks: results.masks,
                        lowResMasks: results.low_res_masks,
                        xtl: Number(results.xtl.data[0]),
                        ytl: Number(results.ytl.data[0]),
                        xbr: Number(results.xbr.data[0]),
                        ybr: Number(results.ybr.data[0]),
                    },
                });
            }).catch((error: unknown) => {
                postMessage({ action: DecoderWorkerAction.DECODE, error: errorToMessage(error) });
            });
        }
    };
}
