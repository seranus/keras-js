import Layer from '../../Layer'
import Tensor from '../../Tensor'

/**
 * PixelShuffler layer class
 * Imported layer bassed on code:
 * by t-ae
 * https://gist.github.com/t-ae/6e1016cc188104d123676ccef3264981
 */
export default class PixelShuffler extends Layer {
    /**
     * Creates an PixelShuffler layer
     *
     * @param {Object} [attrs] - layer config attributes
     * @param {number[]} [attrs.inputs]
     */
    constructor(attrs = {}) {
        super(attrs)
        this.layerClass = 'PixelShuffler'

        const { inputs = [] } = attrs

        //TODO
        //Load config attributes
    }


    /**
    * Layer computational logic
    *
    * @param {Tensor} x
    * @returns {Tensor}
    */
    call(x) {
        if (this.gpu) {
            this._callGPU(x)
        } else {
            this._callCPU(x)
        }
        return this.output
    }

    /**
    * CPU call
    *
    * @param {Tensor} x
    */
    _callCPU(x) {
        //Import flow

        //Check input shape size

        //if data_format

        //unpack input shape

        //mul and div calculations

        //call reshape

        //call permute

        //call reshape

        //else if data_format

        //unpack input shape

        //mul and div calculations

        //call reshape

        //call permute

        //call reshape
    }


    /**
    * GPU call
    *
    * @param {Tensor} x
    */
    _callGPU(x) {
        //TODO
        //Later
    }
}