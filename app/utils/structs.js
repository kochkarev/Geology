const path = require('path');
const fs = require('fs');


class InstAnnotation {
	constructor(imgId) {
		this.imgId = imgId;
		this.instMap = null;
		this.instances = [];
		this.loaded = false;
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

class ImageItem {

	constructor(id, fullFilePath) {
		this.id = id;
	
		this.imagePath = fullFilePath;
		this.imageName = path.parse(fullFilePath).base,
		this.instAnnoPath = this.getAnnoPath(fullFilePath),
	
		this.annoSemanticGT = null;
		this.annoInstGT = new InstAnnotation(id);
		this.annoSemanticPR = null;
		this.annoInstPR = new InstAnnotation(id);
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

class ImageList {
    
    constructor(backend, renderer) {
        this.backend = backend
        this.renderer = renderer
		this.items = new Map();
		this.activeId = -1;
    }

    addImage(fullFilePath) {
		let id = this.items.size + 1;
		let item = new ImageItem(id, fullFilePath);
        this.items.set(id, item);
    }

    addImages(fullFilePaths) {
        for (let fullFilePath of fullFilePaths) {
            this.addImage(fullFilePath)
        }
    }
	
	onAnnotationLoaded(imgId, source) {
		console.log(`inst annotation for image ${imgId} from ${source} received!`);
		let imgStruct = this.items.get(imgId);
		switch (source) {
			case 'GT':
				imgStruct.annoInstGT.setLoaded();
				break;
			case 'PR':
				imgStruct.annoInstPR.setLoaded();
				break;
		}
		if (imgStruct.id === this.activeId) {
			this._sendAnnoToRenderer(imgStruct);
		}
	}

	onActiveImageUpdate(id) {
		this.activeId = id;
		console.log(`active image update: ${this.activeId}`);
		let imgStruct = this.items.get(this.activeId);
		if (!imgStruct.annoInstGT.isLoaded()) {
			this.backend.getInstAnno(imgStruct.instAnnoPath, imgStruct.id, 'GT');
		} else {
			this._sendAnnoToRenderer(imgStruct);
		}
	}

	_sendAnnoToRenderer(imgStruct) {
		this.renderer.send('anno-loaded', imgStruct.annoInstGT);
	}

	updateAnnoInst(inst) {
		this.items.get(inst.imgid)?.updateAnnoInst(inst);
	}

	updateAnnoInstMap(instMap) {
		this.items.get(instMap.imgid)?.updateAnnoInstMap(instMap);
	}

	predict() {
		let imgStruct = this.items.get(this.activeId);
		if (imgStruct.prediction === null) {
			this.backend.predict(imgStruct.imagePath, imgStruct.id);
		}
	}
}


module.exports = {InstdAnnotation: InstAnnotation, ImageList: ImageList}