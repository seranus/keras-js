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

        if (x.tensor.shape.length !== 4 && x.tensor.shape.length  !== 3) {
            throw new Error("Input shape length invalid: " + x.length)
        }

        let batch_size, c, h, w

        if (this.dataFormat === 'channels_first') {
            if (x.tensor.shape.length === 3) {
                [c, h, w] = x.tensor.shape
                batch_size = -1
            } else {
                [batch_size, c, h, w] = x.tensor.shape
            }

            let [rh, rw] = this.size;
            let [oh, ow] = [h * rh, w * rw];
            let oc = math.floor(x / (rh * rw)) //integer division, JS doesn't have int division, slower preformace compared to native int division 
            //out = K.reshape(inputs, (batch_size, h, w, rh, rw, oc))
            //out = K.permute_dimensions(out, (0, 1, 3, 2, 4, 5))
            //out = K.reshape(out, (batch_size, oh, ow, oc))

            //let out = 

        } else if (this.dataFormat === 'channels_last') {
            if (x.tensor.shape.length === 3) {
                [h, w, c] = x.tensor.shape
                batch_size = -1
            } else {
                [batch_size, c, h, w] = x.tensor.shape
            }

            let [rh, rw] = this.size;
            let [oh, ow] = [h * rh, w * rw];
            let oc = math.floor(x / (rh * rw))
        }

        this.output = x;
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