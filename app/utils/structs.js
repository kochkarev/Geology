const path = require('path');
const fs = require('fs');
const UPNG = require('upng-js');
const sizeOf = require('image-size');


class SemAnnotation {
	constructor(name, imgId, data, w, h) {
		this.name = name;
		this.imgId = imgId;
		this.data = data;
		this.w = w;
		this.h = h;
	}
}


class InstAnnotation {
	constructor(name, imgId, w, h) {
		this.name = name;
		this.imgId = imgId;
		this.instMap = null;
		this.instances = [];
		this.loaded = false;
		this.w = w;
		this.h = h;
	}

	addInst(inst) {
		this.instances.push(inst);
	}

	updateInstMap(instMap) {
		this.instMap = instMap;
	}

	isEmpty() {
		return this.instances.length === 0;
	}

	setLoaded() {
		this.loaded = true;
	}

	isLoaded() {
		return this.loaded;
	}

}

class XImage {

	constructor(id, fullFilePath) {
		this.id = id;
		this.w = 0;
		this.h = 0;
	
		this.imagePath = fullFilePath;
		this.imageName = path.parse(fullFilePath).base,
		this.maskPath = this.getAnnoPath(fullFilePath),
	
		this.annoInst = new Map();
		this.annoSemantic = new Map();

		let dims = sizeOf(this.imagePath);
		this.w = dims.width;
		this.h = dims.height;

		this.loadMask(this.maskPath, 'GT');
	}

	loadMask(maskPath, source) {
		if (maskPath != null) {
			let mask = UPNG.decode(fs.readFileSync(maskPath));
			if (mask.width !== this.w || mask.height !== this.h) {
				console.error('incompatible mask size');
				return;
			}
			this.w = mask.width;
			this.h = mask.height;
			var sem = new SemAnnotation(source, this.id, mask.data, this.w, this.h);
			this.annoSemantic.set(source, sem);
		}
	}

	getAnnoPath(imageFullPath) {
		let p = path.parse(imageFullPath);
		let maskFullPath = path.join(p.dir, p.name + '_mask' + '.png');
		try {
			return fs.existsSync(maskFullPath) ? maskFullPath : null;
		} catch (e) {
			return null;
		}
	}

	getAnnoInst(src) {
		if (!this.annoInst.has(src)) {
			this.annoInst.set(src, new InstAnnotation('GT', this.id, this.w, this.h));
		}
		return this.annoInst.get(src);
	}

	updateAnnoInst(inst) {
		this.getAnnoInst(inst.src).addInst(inst);
	}

	updateAnnoInstMap(instMap) {
		this.getAnnoInst(instMap.src).updateInstMap(instMap);
	}

	updateAnnoSem(anno) {
		let a = new SemAnnotation(anno.src, anno.id, anno.data, anno.w, anno.h);
		this.annoSemantic.set(anno.src, a);
	}

}

class XImageCollection {
    
    constructor(backend, renderer) {
        this.backend = backend
        this.renderer = renderer
		this.items = new Map();
		this.activeId = -1;
    }

    addFromPath(fullFilePath) {
		let id = this.items.size + 1;
		let item = new XImage(id, fullFilePath);
        this.items.set(id, item);
    }

    addFromPaths(fullFilePaths) {
        for (let fullFilePath of fullFilePaths) {
            this.addFromPath(fullFilePath)
        }
    }
	
	onAnnotationLoaded(xId, source) {
		console.log(`inst annotation for image ${xId} from ${source} received!`);
		let x = this.items.get(xId);
		x.annoInst.get(source).setLoaded();
		this.sendUpdate(x);
	}

	onActiveImageUpdate(id) {
		this.activeId = id;
		let x = this.items.get(this.activeId);
		if (x.annoSemantic.has('GT') && !x.annoInst.has('GT')) {
			this.backend.requestInstAnno(x.maskPath, x.id, 'GT');
		} else {
			this.sendUpdate(x);
		}
	}

	onModelLoaded() {
		this.renderer.send('model-loaded');
	}

	sendUpdate(x) {
		this.renderer.send('ximage-update', x);
	}

	updateAnnoInst(inst) {
		this.items.get(inst.imgid)?.updateAnnoInst(inst);
	}

	updateAnnoInstMap(instMap) {
		this.items.get(instMap.imgid)?.updateAnnoInstMap(instMap);
	}

	updateAnnoSem(anno) {
		let x = this.items.get(anno.id);
		x.updateAnnoSem(anno);
		this.sendUpdate(x);
	}

	predict() {
		let x = this.items.get(this.activeId);
		if (x.prediction == null) {
			this.backend.predict(x.imagePath, x.id, x.maskPath);
		}
	}
}


module.exports = {InstAnnotation: InstAnnotation, SemAnnotation: SemAnnotation, XImageCollection: XImageCollection}