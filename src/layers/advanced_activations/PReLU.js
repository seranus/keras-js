import Layer from '../../Layer'
import Tensor from '../../Tensor'
import { webgl2 } from '../../WebGL2'
import cwise from 'cwise'
import programSource from './PReLU.glsl'
import tile from 'ndarray-tile'

/**
 * PReLU advanced activation layer class
 *
 * reference code:
 * ```
 * pos = K.relu(x)
 * neg = self.alpha * (x - abs(x)) * 0.5
 * return pos + neg
 * ```
 */
export default class PReLU extends Layer {
  /**
   * Creates a PReLU activation layer
   *
   * @param {Object} [attrs] - layer config attributes
   */
  constructor(attrs = {}) {
    super(attrs)
    this.layerClass = 'PReLU'

    // Layer weights specification
    this.params = ['alpha']

    // GPU setup
    if (this.gpu) {
      this.program = webgl2.compileProgram(programSource)
    }
  }

  _compute = cwise({
    args: ['array', 'array'],
    body: function(_x, alpha) {
      _x = Math.max(_x, 0) + alpha * Math.min(_x, 0)
    }
  })

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
    this.output = x

    //https://github.com/transcranial/keras-js/issues/81
    //Shared axsis kod
    let alpha_tiled;
    let tileAlphas = false;
    let tiling = [];

    for (let i = 0; i < x.tensor.shape.length; i++) {
      if (x.tensor.shape[i] != this.weights['alpha'].tensor.shape[i]) {
        tileAlphas = true;
        tiling.push(x.tensor.shape[i]);
      } else {
        tiling.push(1);
      }
    }
    if (tileAlphas) {
      alpha_tiled = tile(this.weights['alpha'].tensor, tiling);
    } else {
      alpha_tiled = this.weights['alpha'].tensor;
    }

    this._compute(this.output.tensor, alpha_tiled)
  }

  /**
   * GPU call
   *
   * @param {Tensor} x
   */
  _callGPU(x) {
    if (!x.glTexture && !x.glTextureFragments) {
      x.createGLTexture({ type: '2d', format: 'float', supportsTextureFragments: true })
    }

    if (!this.output) {
      this.output = new Tensor([], x.glTextureShape)
      this.output.createGLTexture({ type: '2d', format: 'float', supportsTextureFragments: true })
      if (x.is1D) {
        this.output.is1D = x.is1D
      } else if (x.is2DReshaped || x.is2DSquareReshaped) {
        if (x.is2DReshaped) {
          this.output.is2DReshaped = x.is2DReshaped
        } else if (x.is2DSquareReshaped) {
          this.output.is2DSquareReshaped = x.is2DSquareReshaped
        }
        this.output.originalShape = x.originalShape
        this.output.indicesForReshaped = x.indicesForReshaped
      }
    }

    webgl2.runProgram({
      program: this.program,
      output: this.output,
      inputs: [{ input: x, name: 'x' }, { input: this.weights['alpha'], name: 'alpha' }],
      supportsTextureFragments: true
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
