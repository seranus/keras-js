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
            //this._callGPU(x)
            this._callCPU(x)
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
            let [c, h, w] = x.tensor.shape

            let [rh, rw] = this.size
            let [oh, ow] = [h * rh, w * rw]
            let oc = Math.floor(c / (rh * rw))

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
            let [h, w, c] = x.tensor.shape

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
    * Creates row/col index mappings to map input texture to output texture
    */
    _createIndexMap() {
        if (this.indexMap) {
            return
        }

        const indices = new Tensor([], this.inputShape, { type: Int32Array })
        const indicesRow = new Tensor([], this.inputShape, { type: Int32Array })
        const indicesCol = new Tensor([], this.inputShape, { type: Int32Array })

        if (this.inputShape.length === 2) {
            for (let i = 0; i < this.inputShape[0]; i++) {
                ops.assigns(indicesRow.tensor.pick(i, null), i)
            }
        } else if (this.inputShape.length === 3) {
            for (let i = 0; i < this.inputShape[0]; i++) {
                for (let j = 0; j < this.inputShape[1]; j++) {
                    ops.assigns(indicesRow.tensor.pick(i, j, null), i * this.inputShape[1] + j)
                }
            }
        } else if (this.inputShape.length === 4) {
            for (let i = 0; i < this.inputShape[0]; i++) {
                for (let j = 0; j < this.inputShape[1]; j++) {
                    for (let k = 0; k < this.inputShape[2]; k++) {
                        ops.assigns(
                            indicesRow.tensor.pick(i, j, k, null),
                            i * this.inputShape[1] * this.inputShape[2] + j * this.inputShape[2] + k
                        )
                    }
                }
            }
        }
        for (let c = 0; c < _.last(this.inputShape); c++) {
            ops.assigns(indicesCol.tensor.pick(...Array(this.inputShape.length - 1).fill(null), c), c)
        }
        // i * cols + j
        ops.muls(indices.tensor, indicesRow.tensor, _.last(this.inputShape))
        ops.addeq(indices.tensor, indicesCol.tensor)

        this.indexMap = new Tensor([], this.targetShape, { type: Int32Array })
        this.indexMap.replaceTensorData(new Int32Array(indices.tensor.data))
        if (this.targetShape.length > 2) {
            this.indexMap.reshapeTo2D()
        }

        this.indexMap.createGLTexture({ type: '2d', format: 'int' })
    }

    /**
    * GPU call
    *
    * @param {Tensor} x
    */
    _callGPU(x) {
        if (!x.glTexture) {
            this.inputShape = x.tensor.shape
            if (x.tensor.shape.length <= 2) {
                x.createGLTexture({ type: '2d', format: 'float' })
            } else if (x.tensor.shape.length > 2 && !x.is2DReshaped) {
                x.reshapeTo2D()
                x.createGLTexture({ type: '2d', format: 'float' })
            }
        } else if (x.is2DReshaped || x.is2DSquareReshaped) {
            this.inputShape = x.originalShape
        } else {
            this.inputShape = x.tensor.shape
        }
        this._createIndexMap()

        //copied from cpu, work in progress dont use
        if (!this.output) {
            if (this.dataFormat === 'channels_first') {
                let [c, h, w] = x.tensor.shape

                let [rh, rw] = this.size
                let [oh, ow] = [h * rh, w * rw]
                let oc = Math.floor(c / (rh * rw))

                //TODO optimize
                //reshape
                let out_1 = new Tensor([], [rh, rw, oc, h, w])
                out_1.reshapeTo2D()

                //permute
                let dims = [2, 3, 0, 4, 1]
                const out_2_Shape = dims.map(i => out_1.tensor.shape[i])
                let out_2 = new Tensor([], out_2_Shape)
                out_2.reshapeTo2D()

                //reshape
                this.output = new Tensor([], [oc, oh, ow])
                this.output.reshapeTo2D()

            } else if (this.dataFormat === 'channels_last') {
                let [h, w, c] = x.tensor.shape

                let [rh, rw] = this.size
                let [oh, ow] = [h * rh, w * rw]
                let oc = Math.floor(c / (rh * rw))

                //TODO optimize
                //reshape
                let out_1 = new Tensor([], [h, w, rh, rw, oc])
                out_1.output.reshapeTo2D()

                //permute
                let dims = [0, 2, 1, 3, 4]
                const out_2_Shape = dims.map(i => out_1.tensor.shape[i])
                let out_2 = new Tensor([], out_2_Shape)
                out_2.reshapeTo2D()

                //reshape
                this.output = new Tensor([], [oh, ow, oc])
                this.output.reshapeTo2D()
            }

            this.output.createGLTexture({ type: '2d', format: 'float' })
        }

        //map not getting created after every input
        webgl2.runProgram({
            program: this.mapInputProgram,
            output: this.out_1,
            inputs: [{ input: x, name: 'x' }, { input: this.indexMap, name: 'indexMap' }],
            uniforms: [{ value: x.glTextureShape[1], type: 'int', name: 'inputCols' }]
        })
        webgl2.runProgram({
            program: this.mapInputProgram,
            output: this.out_2,
            inputs: [{ input: x, name: 'x' }, { input: this.indexMap, name: 'indexMap' }],
            uniforms: [{ value: out_1.glTextureShape[1], type: 'int', name: 'inputCols' }]
        })
        webgl2.runProgram({
            program: this.mapInputProgram,
            output: this.output,
            inputs: [{ input: x, name: 'x' }, { input: this.indexMap, name: 'indexMap' }],
            uniforms: [{ value: out_2.glTextureShape[1], type: 'int', name: 'inputCols' }]
        })

        // GPU -> CPU data transfer
        if (this.outbound.length === 0) {
            this.output.transferFromGLTexture()
            if (this.output.is2DReshaped) {
                this.output.reshapeFrom2D()
            } else if (this.output.is2DSquareReshaped) {
                this.output.reshapeFrom2DSquare()
            }
        }
    }
}