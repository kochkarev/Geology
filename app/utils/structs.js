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

	getInstByCoords(x, y) {
		if (!this.instMap) {
			return null;
		}
		let inst_id = this.instMap.data[y * this.instMap.h + x];
		return this.instances[inst_id];
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
    }

    addImage(fullFilePath) {
        let imageItem = {
			id: this.items.length,
            imageFullPath: fullFilePath,
            imageName: path.parse(fullFilePath).base,
			annotationFullPath: this.readAnnotation(fullFilePath),
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
	
	readAnnotation(imageFullPath) {
		let p = path.parse(imageFullPath);
		let maskFullPath = path.join(p.dir, p.name + '_mask' + '.png');
		try{
			return fs.existsSync(maskFullPath) ? maskFullPath : null;
		} catch (e) {
			return null;
		}
	}

	onAnnotationLoaded(imgId) {
		console.log(`annotation for image ${imgId} received!`);
		this.items[imgId].annotation.setLoaded();
	}

	onActiveImageUpdate(activeIdx) {
		console.log(`active image update: ${activeIdx}`);
		let imgStruct = this.items[activeIdx];
		if (!imgStruct.annotation.isLoaded()) {
			this.backend.getFullAnnotation(imgStruct.annotationFullPath, imgStruct.id);
		}
		
	}

	updateAnnoInst(inst) {
		this.items[inst.imgid].annotation.addInst(inst);
	}

	updateAnnoInstMap(instMap) {
		this.items[instMap.imgid].annotation.updateInstMap(instMap);
	}
}


module.exports = {InstdAnnotation: InstAnnotation, ImageList: ImageList}