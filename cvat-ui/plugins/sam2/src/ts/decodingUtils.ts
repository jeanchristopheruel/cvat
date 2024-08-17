// decodingUtils.ts

import { Tensor } from 'onnxruntime-web';

/**
 * Prepare the inputs for the decoder by organizing the embeddings, point coordinates, and point labels.
 * This function returns the necessary tensors for the decoder.
 *
 * @param encoderOutputs - The output embeddings from the encoder model.
 * @param pointCoords - Tensor for point coordinates (user clicks).
 * @param pointLabels - Tensor for point labels (positive/negative clicks).
 * @returns {DecodeBody} - An object containing the prepared inputs for the decoder.
 */
export function prepareDecodingInputs(
    encoderOutputs: any,
    pointCoords: Tensor,
    pointLabels: Tensor
): any {
    const { image_embed, high_res_feats_0, high_res_feats_1 } = encoderOutputs;

    return {
        image_embed,
        high_res_feats_0,
        high_res_feats_1,
        point_coords: pointCoords,
        point_labels: pointLabels,
        // Provide a default empty mask input (all zeros)
        mask_input: new Tensor('float32', new Float32Array(256 * 256), [1, 1, 256, 256]),
        // Indicate whether a mask input is provided (1 if true, 0 if false)
        has_mask_input: new Tensor('float32', new Float32Array([0]), [1]),
    };
}
