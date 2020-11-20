const { app, BrowserWindow, ipcMain, Menu, dialog } = require('electron');
const { format } = require('url');
const path = require('path');
const { BackendCommunicator } = require('./utils/backend');
const { ImageList } = require('./utils/structs');

const backend_cfg = {
	pyenv_name: 'tf',
	pyenv_path: 'C:/Users/xubiker/Anaconda3/envs/tf/python',
	src_path: './backend/server.py'
};

let backend = null;
let imageList = null;


function handler_array(arr, header) {
	if ('ext' in header) {
		if (header.ext == 'chunk') {
			let chunk = {'id': header.id, 'x': header.x, 'y': header.y, 'w': header.shape[1], 'h': header.shape[0], 'class': header.class, 'imgid': header.imgid, 'mask': arr};
			imageList.updateAnnotationChunk(chunk);
		} else if (header.ext == 'chunk-map') {
			let chunkMap = {'w': header.shape[1], 'h': header.shape[0], 'imgid': header.imgid, 'data': arr}
			imageList.updateAnnotationChunkMap(chunkMap);
		}
	} else {
		console.log(`#arr: shape[${header.shape}]. Got ${arr.length} bytes`);
	}
	//this.win.webContents.send('anno-update', arr);
}

function handler_signal(s) {
	if (s === 'A1') {
		console.log('annotation received!');
		imageList.items[0].annotation.getChunkByCoords(100, 500);
	}
}

function handler_string(s) {
	console.log('#str: ' + s);
}


function foo(x, y) {
	let chunk = imageList?.getActiveItem()?.annotation.getChunkByCoords(x, y);
	if (chunk) {
		console.log(chunk.id);
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
			backend_cfg,
			(arr, header) => handler_array(arr, header),
			(sig) => handler_signal(sig),
			(str) => handler_string(str)
		);
		imageList = new ImageList(backend, win.webContents);


		backend.ping();
		backend.ping_image();

		// ipcMain.on('stop-algo', (event, arg) => backend.stop_algo());
		// // ipcMain.on('btn_req_click', (event, arg) => backend.ping_image());
		// ipcMain.on('anno', (event, args) => backend.send_annotation(args[0], args[1]));
		// ipcMain.on('mouse-move', (event, args) => mouseHandler.mouseMoveUpdate(args[0], args[1]));
	});

	// ipcMain.on('stop-algo', (event, arg) => backend.stop_algo());
	// ipcMain.on('btn_req_click', (event, arg) => backend.ping_image());
    // ipcMain.on('anno', (event, args) => backend.send_annotation(args[0], args[1]));
    ipcMain.on('mouse-move', (event, args) => foo(args[0], args[1]));

	const mainMenu = Menu.buildFromTemplate(menuTemplate);
	Menu.setApplicationMenu(mainMenu);

	//win.webContents.openDevTools();    
	
	
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
				imageList.changeActiveImageIdx();
			}
		})
		.catch( (e) =>
			console.log('promise rejected: ' + e)
		);
}