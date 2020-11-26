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

class ImageList {
    
    constructor(backend, renderer) {
        this.backend = backend
        this.renderer = renderer
		this.items = new Map();
		this.activeId = -1;
    }

    addImage(fullFilePath) {
		let imgId = this.items.size + 1;
        let imageItem = {
			id: imgId,
            imagePath: fullFilePath,
            imageName: path.parse(fullFilePath).base,
			instAnnoPath: this.getAnnoPath(fullFilePath),
			instAnno: new InstAnnotation(imgId)
		};
        this.items.set(imgId, imageItem);
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
		let imgStruct = this.items.get(imgId);
		imgStruct.instAnno.setLoaded();
		if (imgStruct.id === this.activeId) {
			this._sendAnnoToRenderer(imgStruct);
		}
	}

	onActiveImageUpdate(id) {
		this.activeId = id;
		console.log(`active image update: ${this.activeId}`);
		let imgStruct = this.items.get(this.activeId);
		if (!imgStruct.instAnno.isLoaded()) {
			this.backend.getInstAnno(imgStruct.instAnnoPath, imgStruct.id);
		} else {
			this._sendAnnoToRenderer(imgStruct);
		}
	}

	_sendAnnoToRenderer(imgStruct) {
		this.renderer.send('anno-loaded', imgStruct.instAnno);
	}

	updateAnnoInst(inst) {
		this.items.get(inst.imgid)?.instAnno.addInst(inst);
	}

	updateAnnoInstMap(instMap) {
		this.items.get(instMap.imgid)?.instAnno.updateInstMap(instMap);
	}
}


module.exports = {InstdAnnotation: InstAnnotation, ImageList: ImageList}