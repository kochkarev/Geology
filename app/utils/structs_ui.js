class ImgStructWrapper {
    constructor(imgStruct) {
        this.imgStruct = imgStruct;
        let li = document.createElement('li');
        li.className = "list-group-item";
        li.innerHTML = imgStruct.imageName;
        this.li = li;
    }
}

class ImageListWrapper {
    
    constructor(listGroupHTML, updateImageCallBack) {
        this.items = [];
        this.activeIdx = 0;
        this.activeIdxPrev = 0;
        this.listGroupHTML = listGroupHTML;
        this.updateImageCallBack = updateImageCallBack;
    }

    add(imgStruct) {
        let wrapper = new ImgStructWrapper(imgStruct);
        this.items.push(wrapper);
        this.listGroupHTML.append(wrapper.li);
        if (this.items.length === 1) {
            this.selectByIndex(0);
        }
    }

    addMany(imgStructs) {
        for (const [key, imgStruct] of imgStructs.entries()) {
            this.add(imgStruct);
        }
    }

    moveSelection(code) {
        let n = this.items.length;
        if (n === 0)
            return;
        let idx = 0;
        if (code === 'ArrowDown')
            idx = (this.activeIdx + 1) % n;
        if (code === 'ArrowUp')
            idx = (n + this.activeIdx - 1) % n;
        this.selectByIndex(idx);
    }

    selectClick(e) {
        if (this.items.length == 0)
            return;
        let li = e.target.closest('li');
        let idx = Array.from(this.listGroupHTML.children).indexOf(li);
        this.selectByIndex(idx);
    }

    selectByIndex(newIndex) {
        this.activeIdxPrev = this.activeIdx;
        this.activeIdx = newIndex;
        this.listGroupHTML.children[this.activeIdxPrev].classList.remove('active');
        this.listGroupHTML.children[this.activeIdx].classList.add('active');
        this.updateImageCallBack(this.items[this.activeIdx].imgStruct);
    }
};

module.exports = {ImgStructWrapper: ImgStructWrapper, ImageListWrapper: ImageListWrapper};