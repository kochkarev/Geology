const { app, BrowserWindow, ipcMain, Menu, dialog } = require('electron');
const { format } = require('url');
const path = require('path');
const { BackendCommunicator } = require('./utils/backend');
const { ImageList } = require('./utils/structs');

const backendCfg = {
	pyEnvName: 'tf',
	pyEnvPath: 'C:/Users/xubiker/Anaconda3/envs/tf/python',
	srcPath: './backend/server.py'
};

let backend = null;
let imageList = null;
let annoController = null;


function handlerArray(arr, header) {
	if ('ext' in header) {
		if (header.ext == 'inst') {
			let inst = {
				'id': header.id, 'x': header.x, 'y': header.y, 'w': header.shape[1], 'h': header.shape[0],
				'class': header.class, 'imgid': header.imgid, 'mask': arr
			};
			imageList.updateAnnoInst(inst);
		} else if (header.ext == 'inst-map') {
			let instMap = {'w': header.shape[1], 'h': header.shape[0], 'imgid': header.imgid, 'data': arr}
			imageList.updateAnnoInstMap(instMap);
		}
	} else {
		console.log(`#arr: shape[${header.shape}]. Got ${arr.length} bytes`);
	}
	//this.win.webContents.send('anno-update', arr);
}

function handlerSignal(s) {
	if (s.startsWith('A')) {
		let imgId = parseInt(s.slice(1));
		// console.log(`annotation for image ${imgId} received!`);
		imageList.onAnnotationLoaded(imgId);
	}
}

function handlerString(s) {
	console.log('#str: ' + s);
}


class AnnotattionRendererController {
	constructor(imgList, renderer) {
		this.imgList = imgList;
		this.renderer = renderer;
		this.imgId = null;
		this.prevIid = null;
	}

	onMouseMove(x, y) {
		// let inst = imageList?.getActiveItem()?.annotation.getInstByCoords(x, y);
		// if (inst) {
			// console.log(inst.id);
		// }
	}

	onActiveImageUpdate() {

	}
}

function foo(x, y) {
	let inst = imageList?.getActiveItem()?.annotation.getInstByCoords(x, y);
	if (inst) {
		console.log(inst.id);
	}
}

app.on('ready', () => {
	win = new BrowserWindow({ width: 1400, height: 800, webPreferences: { nodeIntegration: true, enableRemoteModule: false}})
	win.loadURL(format({
		pathname: path.join(__dirname, 'index.html'),
		protocol: 'file:',
		slashes: true,
	}))

	win.webContents.on('did-finish-load', () => {
        backend = new BackendCommunicator(
			backendCfg,
			(arr, header) => handlerArray(arr, header),
			(sig) => handlerSignal(sig),
			(str) => handlerString(str)
		);
		imageList = new ImageList(backend, win.webContents);

		annoController = new AnnotattionRendererController(imageList, win.webContents);


		backend.ping();
		backend.pingImage();

		// ipcMain.on('stop-algo', (event, arg) => backend.stop_algo());
		// // ipcMain.on('btn_req_click', (event, arg) => backend.ping_image());
		// ipcMain.on('anno', (event, args) => backend.send_annotation(args[0], args[1]));
		// ipcMain.on('mouse-move', (event, args) => mouseHandler.mouseMoveUpdate(args[0], args[1]));
	});

	// ipcMain.on('stop-algo', (event, arg) => backend.stop_algo());
	// ipcMain.on('btn_req_click', (event, arg) => backend.ping_image());
    // ipcMain.on('anno', (event, args) => backend.send_annotation(args[0], args[1]));
	ipcMain.on('active-image-update', (event, args) => imageList.onActiveImageUpdate(args));
	// ipcMain.on('mouse-move', (event, args) => annoController.onMouseMove(args[0], args[1]));

	const mainMenu = Menu.buildFromTemplate(menuTemplate);
	Menu.setApplicationMenu(mainMenu);

	win.webContents.openDevTools();    
	
	
})

app.on('before-quit', () => {
	backend.stop();
});

const menuTemplate = [
	{
		label: 'Project',
		submenu: [
		{
			label: 'Open'
		},
		{
			label: 'Import images',
			accelerator: 'CmdOrCtrl+O',
			click: () => {dialog_import_files()}
		},
		{
			label: 'Save'
		},
		{
			type: 'separator'
		},
		{
			label: 'Quit',
			accelerator: process.platform == 'darwin' ? 'Command+Q' : 'Ctrl+Q',
			click: () => {app.quit()}
		}
		],
	},
	{
		label: 'Edit',
		submenu: [
			
		]
	},
	{
		label: 'Help'
	}
]


function dialog_import_files() {
	dialog.showOpenDialog({
		title: 'Select images to import...',
		properties: ['openFile', 'multiSelections'],
		defaultPath: __dirname + "\\sample_data",
		buttonLabel: "Select",
		filters: [
		    { name: 'Images', extensions: ['jpg']}
		]
		})
		.then(res => {
			if (!res.canceled) {
				imageList.addImages(res.filePaths);
				win.webContents.send('files-added', res.filePaths);
				// imageList.changeActiveImageIdx();
			}
		})
		.catch( (e) =>
			console.log('promise rejected: ' + e)
		);
}