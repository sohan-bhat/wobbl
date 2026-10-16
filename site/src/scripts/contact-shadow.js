// A soft contact shadow under a scene's subject, on the floor or bench it stands on, as three.js's contact
// shadows example and model-viewer's shadow draw one: the subject is seen from below the floor with an
// orthographic camera, darker the nearer it is to the floor, blurred twice and laid on the floor as a
// see through picture. It is fake (nothing shadows itself) and cheap, one small render and six blur passes
// whenever the scene moves, so it runs on phones and in the stills drawn at build time.
import {
  Color, DoubleSide, Mesh, MeshBasicMaterial, MeshDepthMaterial, OrthographicCamera, PlaneGeometry, Scene, ShaderMaterial,
  WebGLRenderTarget,
} from 'three';

const SIZE = 256;

const blurMaterial = () => new ShaderMaterial({
  depthTest: false, depthWrite: false,
  uniforms: { tMap: { value: null }, uStep: { value: [0, 0] } },
  vertexShader: 'varying vec2 vUv; void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }',
  fragmentShader: /* glsl */ `
    uniform sampler2D tMap;
    uniform vec2 uStep;
    varying vec2 vUv;
    void main() {
      vec4 s = texture2D(tMap, vUv) * 0.2270270270;
      s += (texture2D(tMap, vUv + uStep * 1.3846153846) + texture2D(tMap, vUv - uStep * 1.3846153846)) * 0.3162162162;
      s += (texture2D(tMap, vUv + uStep * 3.2307692308) + texture2D(tMap, vUv - uStep * 3.2307692308)) * 0.0702702703;
      gl_FragColor = s;
    }`,
});

/**
 * casters: the objects that cast it. center and size: the square of floor it covers, in the scene's units,
 * at height floor. reach: how far above the floor a caster still darkens it. Returns the floor's mesh to add
 * to the scene and render(renderer, scene), which redraws the shadow from the casters as they are now.
 */
export function contactShadow({ casters, center, size, floor, reach, opacity = 0.62 }) {
  const a = new WebGLRenderTarget(SIZE, SIZE), b = new WebGLRenderTarget(SIZE, SIZE);
  const half = size / 2;
  // from a little under the floor looking up, so a face resting on the floor is in front of the camera and
  // what is nearest the floor comes out darkest
  const below = reach * 0.04;
  const cam = new OrthographicCamera(-half, half, half, -half, 0, reach + below);
  cam.up.set(0, 1, 0);
  cam.position.set(center[0], center[1], floor - below);
  cam.lookAt(center[0], center[1], floor + 1);
  cam.updateMatrixWorld();
  const depth = new MeshDepthMaterial();
  depth.depthTest = false;
  depth.depthWrite = false;
  depth.side = DoubleSide;
  depth.onBeforeCompile = (shader) => {
    shader.fragmentShader = shader.fragmentShader.replace('gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );',
      'gl_FragColor = vec4( vec3( 0.0 ), pow( 1.0 - fragCoordZ, 1.6 ) );');
  };
  const plane = new Mesh(new PlaneGeometry(size, size), new MeshBasicMaterial({
    map: a.texture, transparent: true, opacity, depthWrite: false, side: DoubleSide,
  }));
  // the camera looked up, so its picture is the floor seen from below: mirrored across x
  plane.scale.x = -1;
  plane.position.set(center[0], center[1], floor);
  plane.renderOrder = -1;
  plane.frustumCulled = false;
  const quad = new Scene(), blur = blurMaterial(), q = new Mesh(new PlaneGeometry(2, 2), blur);
  q.frustumCulled = false;
  quad.add(q);

  function pass(src, dst, dx, dy, renderer) {
    blur.uniforms.tMap.value = src.texture;
    blur.uniforms.uStep.value = [dx / SIZE, dy / SIZE];
    renderer.setRenderTarget(dst);
    renderer.clear();
    renderer.render(quad, cam);
  }

  function render(renderer, scene) {
    const hidden = [];
    scene.traverse((o) => {
      if (o.isMesh || o.isLineSegments || o.isLine || o.isInstancedMesh) {
        if (o.visible && !casters.has(o)) { hidden.push(o); o.visible = false; }
      }
    });
    const wasScissor = renderer.getScissorTest(), wasColor = renderer.getClearColor(new Color()), wasAlpha = renderer.getClearAlpha();
    renderer.setScissorTest(false);
    renderer.setClearColor(0x000000, 0);
    const over = scene.overrideMaterial;
    scene.overrideMaterial = depth;
    renderer.setRenderTarget(a);
    renderer.clear();
    renderer.render(scene, cam);
    scene.overrideMaterial = over;
    for (const o of hidden) o.visible = true;
    // blurred well past the footprint, since the footprint itself sits under the subject: what shows is the
    // soft dark halo round where it rests, as ambient occlusion would give
    for (const st of [4.5, 2.5, 1.2]) {
      pass(a, b, st, 0, renderer);
      pass(b, a, 0, st, renderer);
    }
    renderer.setRenderTarget(null);
    renderer.setScissorTest(wasScissor);
    renderer.setClearColor(wasColor, wasAlpha);
  }

  function dispose() {
    a.dispose(); b.dispose(); depth.dispose(); blur.dispose();
    plane.geometry.dispose(); plane.material.dispose(); q.geometry.dispose();
  }
  return { plane, render, dispose };
}
