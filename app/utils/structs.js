const path = require('path');
const fs = require('fs');
const UPNG = require('upng-js');


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
	
		this.annoSemanticGT = this.loadMask(this.maskPath); // this updates w, h
		this.annoInstGT = new InstAnnotation('GT', id, this.w, this.h);
		this.annoSemanticPR = null;
		this.annoInstPR = new InstAnnotation('PR', id, this.w, this.h);
	}

	loadMask(maskPath) {
		let mask = UPNG.decode(fs.readFileSync(maskPath));
		this.w = mask.width;
		this.h = mask.height;
		return new SemAnnotation('GT', this.id, mask.data, this.w, this.h);
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

	updateAnnoInst(inst) {
		switch (inst.src) {
			case 'GT':
				this.annoInstGT.addInst(inst);
				break;
			case 'PR':
				this.annoInstPR.addInst(inst);
				break;
		}
	}

	updateAnnoInstMap(instMap) {
		switch (instMap.src) {
			case 'GT':
				this.annoInstGT.updateInstMap(instMap);
				break;
			case 'PR':
				this.annoInstPR.updateInstMap(instMap);
				break;
		}
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
		switch (source) {
			case 'GT':
				x.annoInstGT.setLoaded();
				break;
			case 'PR':
				x.annoInstPR.setLoaded();
				break;
		}
		if (x.id === this.activeId) {
			this.sendUpdate(x);
		}
	}

	onActiveImageUpdate(id) {
		this.activeId = id;
		let x = this.items.get(this.activeId);
		if (!x.annoInstGT.isLoaded()) {
			this.backend.requestInstAnno(x.maskPath, x.id, 'GT');
		} else {
			this.sendUpdate(x);
		}
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

	predict() {
		let x = this.items.get(this.activeId);
		if (x.prediction === null) {
			this.backend.predict(x.imagePath, x.id);
		}
	}
}


module.exports = {InstAnnotation: InstAnnotation, SemAnnotation: SemAnnotation, XImageCollection: XImageCollection}