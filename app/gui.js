const {ipcRenderer} = require('electron');

const fs = require('fs');
const {ShallowImageList} = require('./utils/structs_ui');
const {ActiveImageWithAnnotationRenderer} = require('./utils/annotation_renderer');

const canvClassAnno = document.getElementById('canv-class-anno');
const mainImageZone = document.getElementById('main-image-zone');

let mouseDown = false;


function getLabelsDecoded(config='./config/labels.json') {
    const jf = JSON.parse(fs.readFileSync(config));
    let labelsDecoded = {};
    for (const [label, className] of Object.entries(jf.LabelsToClasses)) {
        const color = jf.ClassesToColors[className];
        labelsDecoded[label] = {"name": className, "color": color};
    }
    return labelsDecoded;    
}


let R = new ActiveImageWithAnnotationRenderer(
    document.getElementById('img'),
    document.getElementById('canv-anno-tmp'),
    document.getElementById('canv-inst-tmp'),
    document.getElementById('canv-class-anno'),
    document.getElementById('canv-inst-anno'),
    document.getElementById('statistics-text'),
    getLabelsDecoded(),
    scaleCOeff=0.4
);


mainImageZone.addEventListener('mousemove', (e) => {
    let rect = canvClassAnno.getBoundingClientRect();
    let x = e.clientX - rect.left;
    let y = e.clientY - rect.top;
    if (mouseDown) {
    }
    else {
        R.cursorMove(x, y);
    }
});

let imgStructs = new ShallowImageList(
    document.getElementById('files-list'),
    (struct) => R.changeContext(struct)
);

document.addEventListener('keydown', (e) => {
    if (e.code === 'ArrowDown' || e.code === 'ArrowUp')
        imgStructs.moveSelection(e.code);
    else if (e.code === 'BracketLeft' || e.code === 'BracketRight')
        R.changeScale(e.code);    
});

imgStructs.listGroupHTML.addEventListener('click', e => imgStructs.selectClick(e));

document.getElementById('btn_load').addEventListener('click', () => ipcRenderer.send('load-model'));
document.getElementById('btn_predict').addEventListener('click', () => ipcRenderer.send('predict'));
// document.getElementById('btn_stop').addEventListener('click', () => ipcRenderer.send('stop-algo'));

ipcRenderer.on('files-added', (e, data) => {
    imgStructs.addFiles(data);
});

ipcRenderer.on('anno-loaded', (event, anno) => {
    console.log(`Hey! Got instance annotation update ${anno.imgId}`);
    R.instAnnoUpdateFromMain(anno);
});

