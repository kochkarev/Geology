const path = require('path');
const fs = require('fs');

class ChunkedAnnotation {
	constructor() {
		this.chunkMap = null;
		this.chunks = [];
	}

	addChunk(chunk) {
		this.chunks.push(chunk);
	}

	updateChunkMap(chunkMap) {
		this.chunkMap = chunkMap;
	}

	getChunkByCoords(x, y) {
		if (!this.chunkMap) {
			return null;
		}
		let chunk_id = this.chunkMap.data[y * this.chunkMap.h + x];
		return this.chunks[chunk_id];
	}
}

class ImageList {
    
    constructor(backend, renderer) {
        this.backend = backend
        this.renderer = renderer
        this.items = [];
        this.activeIdx = 0;
        this.activeIdxPrev = 0;
    }

    addImage(fullFilePath) {
        let imageItem = {
			id: this.items.length,
            imageFullPath: fullFilePath,
            imageName: path.parse(fullFilePath).base,
			annotationFullPath: this.readAnnotation(fullFilePath),
			annotation: new ChunkedAnnotation()
		};
        this.items.push(imageItem);
    }

    addImages(fullFilePaths) {
        for (let fullFilePath of fullFilePaths) {
            this.addImage(fullFilePath)
        }
        console.log(this.items);
    }

    updateRendererImage() {
        //this.renderer.send();
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

	getActiveItem() {
		return this.items.length === 0 ? null : this.items[this.activeIdx]
	}

	changeActiveImageIdx(newActiveIdx = 0) {
		this.activeIdxPrev = this.activeIdx;
		this.activeIdx = newActiveIdx;
		let imgStruct = this.items[newActiveIdx];
		console.log(imgStruct);
		this.backend.getFullAnnotation(imgStruct.annotationFullPath, imgStruct.id);
	}

	updateAnnotationChunk(chunk) {
		this.items[chunk.imgid].annotation.addChunk(chunk);
	}

	updateAnnotationChunkMap(chunkMap) {
		this.items[chunkMap.imgid].annotation.updateChunkMap(chunkMap);
	}
}


module.exports = {ChunkedAnnotation: ChunkedAnnotation, ImageList: ImageList}