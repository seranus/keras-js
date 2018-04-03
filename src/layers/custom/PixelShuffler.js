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

    // GPU setup
    if (this.gpu) {
      this.mapInputProgram = webgl2.compileProgram(mapInputProgramSource)
    }
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
  _createIndexMapReshape(indexMap) {
    if (indexMap) {
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
      //pixelshuffler needs 5
    } else if (this.inputShape.length === 5) {
      for (let i = 0; i < this.inputShape[0]; i++) {
        for (let j = 0; j < this.inputShape[1]; j++) {
          for (let k = 0; k < this.inputShape[2]; k++) {
            for (let g = 0; g < this.inputShape[3]; g++) {
              ops.assigns(
                indicesRow.tensor.pick(i, j, k, g, null),
                i * this.inputShape[1] * this.inputShape[2] * this.inputShape[3]
                + j * this.inputShape[2] * this.inputShape[3]
                + k * this.inputShape[3]
                + g
              )
            }
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

    indexMap = new Tensor([], this.targetShape, { type: Int32Array })
    indexMap.replaceTensorData(new Int32Array(indices.tensor.data))
    if (this.targetShape.length > 2) {
      indexMap.reshapeTo2D()
    }

    indexMap.createGLTexture({ type: '2d', format: 'int' })
  }

  _createIndexMapPermute() {
    if (this.indexMapPermute) {
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
      //pixelshuffler needs 5
    } else if (this.inputShape.length === 5) {
      for (let i = 0; i < this.inputShape[0]; i++) {
        for (let j = 0; j < this.inputShape[1]; j++) {
          for (let k = 0; k < this.inputShape[2]; k++) {
            for (let g = 0; g < this.inputShape[3]; g++) {
              ops.assigns(
                indicesRow.tensor.pick(i, j, k, g, null),
                i * this.inputShape[1] * this.inputShape[2] * this.inputShape[3]
                + j * this.inputShape[2] * this.inputShape[3]
                + k * this.inputShape[3]
                + g
              )
            }
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

    const outputShape = this.dims.map(i => this.inputShape[i])
    this.indexMapPermute = new Tensor([], outputShape, { type: Int32Array })
    ops.assign(this.indexMap.tensor, indices.tensor.transpose(...this.dims))
    if (outputShape.length > 2) {
      this.indexMapPermute.reshapeTo2D()
    }

    this.indexMapPermute.createGLTexture({ type: '2d', format: 'int' })
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

    let targetShape = []
    let dims = []
    let targetShapeOutput = []

    if (this.dataFormat === 'channels_first') {
      let [c, h, w] = x.tensor.shape

      let [rh, rw] = this.size
      let [oh, ow] = [h * rh, w * rw]
      let oc = Math.floor(c / (rh * rw))

      targetShape = [rh, rw, oc, h, w]
      dims = [2, 3, 0, 4, 1]
      targetShapeOutput = [oc, oh, ow]
    } else if (this.dataFormat === 'channels_last') {
      let [h, w, c] = x.tensor.shape

      let [rh, rw] = this.size
      let [oh, ow] = [h * rh, w * rw]
      let oc = Math.floor(c / (rh * rw))

      targetShape = [h, w, rh, rw, oc]
      dims = [0, 2, 1, 3, 4]
      targetShapeOutput = [oh, ow, oc]
    }

    this.targetShape = targetShape;
    //reshape
    this._createIndexMapReshape(this.inputIndexMap)
    if (!this.output) {
      this.output = new Tensor([], targetShape)
      if (this.targetShape.length > 2) {
        this.output.reshapeTo2D()
      }
      this.output.createGLTexture({ type: '2d', format: 'float' })
    }

    webgl2.runProgram({
      program: this.mapInputProgram,
      output: this.output,
      inputs: [{ input: x, name: 'x' }, { input: this.inputIndexMap, name: 'indexMap' }],
      uniforms: [{ value: x.glTextureShape[1], type: 'int', name: 'inputCols' }]
    })

    x = this.output
    this.inputShape = this.output.tensor.shape
    this.dims = dims

    this.output = undefined

    //permute
    this._createIndexMapPermute()

    if (!this.output) {
      const outputShape = this.dims.map(i => this.inputShape[i])
      this.output = new Tensor([], outputShape)
      if (outputShape.length > 2) {
        this.output.reshapeTo2D()
      }
      this.output.createGLTexture({ type: '2d', format: 'float' })
    }

    webgl2.runProgram({
      program: this.mapInputProgram,
      output: this.output,
      inputs: [{ input: x, name: 'x' }, { input: this.indexMapPermute, name: 'indexMap' }],
      uniforms: [{ value: x.glTextureShape[1], type: 'int', name: 'inputCols' }]
    })

    //reshape
    x = this.output
    this.inputShape = this.output.tensor.shape
    this.targetShape = targetShapeOutput;

    this.output = undefined

    this._createIndexMapReshape(this.outputIndexMap)

    if (!this.output) {
      this.output = new Tensor([], targetShape)
      if (this.targetShape.length > 2) {
        this.output.reshapeTo2D()
      }
      this.output.createGLTexture({ type: '2d', format: 'float' })
    }

    webgl2.runProgram({
      program: this.mapInputProgram,
      output: this.output,
      inputs: [{ input: x, name: 'x' }, { input: this.outputIndexMap, name: 'indexMap' }],
      uniforms: [{ value: x.glTextureShape[1], type: 'int', name: 'inputCols' }]
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