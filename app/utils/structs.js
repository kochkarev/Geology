const path = require('path');
const fs = require('fs');


class InstAnnotation {
	constructor() {
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

class ImageList {
    
    constructor(backend, renderer) {
        this.backend = backend
        this.renderer = renderer
		this.items = [];
		this.activeIdx = -1;
    }

    addImage(fullFilePath) {
        let imageItem = {
			id: this.items.length + 1,
            imageFullPath: fullFilePath,
            imageName: path.parse(fullFilePath).base,
			annotationFullPath: this.getAnnoPath(fullFilePath),
			annotation: new InstAnnotation()
		};
        this.items.push(imageItem);
    }

    addImages(fullFilePaths) {
        for (let fullFilePath of fullFilePaths) {
            this.addImage(fullFilePath)
        }
        console.log(this.items);
    }
	
	getAnnoPath(imageFullPath) {
		let p = path.parse(imageFullPath);
		let maskFullPath = path.join(p.dir, p.name + '_mask' + '.png');
		try{
			return fs.existsSync(maskFullPath) ? maskFullPath : null;
		} catch (e) {
			return null;
		}
	}

	onAnnotationLoaded(imgId) {
		console.log(`inst annotation for image ${imgId} received!`);
		let imgStruct = this.items[imgId - 1];
		imgStruct.annotation.setLoaded();
		if (imgId - 1 === this.activeIdx) {
			this._sendAnnoToRenderer(imgStruct);
		}
	}

	onActiveImageUpdate(id) {
		this.activeIdx = id - 1;
		console.log(`active image update: ${this.activeIdx}`);
		let imgStruct = this.items[this.activeIdx];
		if (!imgStruct.annotation.isLoaded()) {
			this.backend.getInstAnno(imgStruct.annotationFullPath, imgStruct.id);
		} else {
			this._sendAnnoToRenderer(imgStruct);
		}
	}

	_sendAnnoToRenderer(imgStruct) {
		this.renderer.send('anno-loaded', imgStruct.id, imgStruct.annotation);
	}

	updateAnnoInst(inst) {
		this.items[inst.imgid - 1].annotation.addInst(inst);
	}

	updateAnnoInstMap(instMap) {
		this.items[instMap.imgid - 1].annotation.updateInstMap(instMap);
	}
}


module.exports = {InstdAnnotation: InstAnnotation, ImageList: ImageList}