const {ipcRenderer} = require('electron');

const {ShallowImageList} = require('./utils/structs_ui');
const {ActiveImageWithAnnotationRenderer} = require('./utils/annotation_renderer');

const canvClassAnno = document.getElementById('canv-class-anno');
const main_image_zone = document.getElementById('main-image-zone');

let mouseDown = false;

const annoColorsArr = [
    [0, 0, 0],
    [255, 255, 0],
    [0, 255, 255],
    [0, 255, 0],
    [0, 0, 255],
    [255, 0, 0],
    [100, 255, 0],
    [255, 100, 100],
    [100, 255, 255],
    [200, 50, 80],
    [80, 50, 200],
    [20, 120, 255],
    [120, 20, 255],
];


let R = new ActiveImageWithAnnotationRenderer(
    document.getElementById('img'),
    document.getElementById('canv-anno-tmp'),
    document.getElementById('canv-inst-tmp'),
    document.getElementById('canv-class-anno'),
    document.getElementById('canv-inst-anno'),
    annoColorsArr
);


main_image_zone.addEventListener('mousemove', (e) => {
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

document.getElementById('btn_st1').addEventListener('click', () => runClick(step=1));
document.getElementById('btn_st2').addEventListener('click', () => runClick(step=2));
document.getElementById('btn_stop').addEventListener('click', () => ipcRenderer.send('stop-algo'));

ipcRenderer.on('files-added', (e, data) => {
    imgStructs.addFiles(data);
});

ipcRenderer.on('anno-loaded', (event, anno) => {
    console.log(`Hey! Got instance annotation update ${anno.imgId}`);
    R.instAnnoUpdateFromMain(anno);
});

