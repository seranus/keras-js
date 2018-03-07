import Layer from '../../Layer'
import Tensor from '../../Tensor'
import _ from 'lodash'
import ops from 'ndarray-ops'

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
     * @param {number|number[]} [attrs.size] - upsampling factor, int or tuple of int (length 2)
     * @param {string} [attrs.data_format] - either 'channels_last' or 'channels_first'
     */
    constructor(attrs = {}) {
        super(attrs)
        this.layerClass = 'PixelShuffler'

        const { size = [2, 2], data_format = 'channels_last' } = attrs

        if (Array.isArray(size)) {
            this.size = size
        } else {
            this.size = [size, size]
        }

        this.dataFormat = data_format

        this.description = `size ${this.size.join('x')} data format ${this.dataFormat}`
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

        if (x.tensor.shape.length !== 4 && x.tensor.shape.length !== 3) {
            throw new Error("Input shape length invalid: " + x.length)
        }

        //output shape
        //shape[0] expected to be always none
        //none is unsuported
        if (this.dataFormat === 'channels_first') {
            [c, h, w] = x.tensor.shape

            let [rh, rw] = this.size
            let [oh, ow] = [h * rh, w * rw]
            let oc = Math.floor(c / (rh * rw)) //integer division, JS doesn't have int division, slower preformace compared to native

            //TODO optimize
            //reshape
            let out_1 = new Tensor([], [rh, rw, oc, h, w])
            out_1.replaceTensorData(x.tensor.data)

            //permute
            let dims = [2, 3, 0, 4, 1]
            const out_2_Shape = dims.map(i => out_1.tensor.shape[i])
            let out_2 = new Tensor([], out_2_Shape)
            ops.assign(out_2.tensor, out_1.tensor.transpose(...dims))

            //reshape
            this.output = new Tensor([], [oc, oh, ow])
            this.output.replaceTensorData(out_2.tensor.data)

        } else if (this.dataFormat === 'channels_last') {
            [h, w, c] = x.tensor.shape

            let [rh, rw] = this.size
            let [oh, ow] = [h * rh, w * rw]
            let oc = Math.floor(c / (rh * rw))

            //TODO optimize
            //reshape
            let out_1 = new Tensor([], [h, w, rh, rw, oc])
            out_1.replaceTensorData(x.tensor.data)

            //permute
            let dims = [0, 2, 1, 3, 4]
            const out_2_Shape = dims.map(i => out_1.tensor.shape[i])
            let out_2 = new Tensor([], out_2_Shape)
            ops.assign(out_2.tensor, out_1.tensor.transpose(...dims))

            //reshape
            this.output = new Tensor([], [oh, ow, oc])
            this.output.replaceTensorData(out_2.tensor.data)
        }
    }


    /**
    * GPU call
    *
    * @param {Tensor} x
    */
    _callGPU(x) {
        //TODO
        //Later

        this.output = x
    }
}