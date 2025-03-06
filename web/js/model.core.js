// ... existing code ...

function addCustomLabel_EasyOCR(nodeType, nodeData, widgetName = "detect") {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        const node = this;
        const detectWidget = node.widgets.find((w) => w.name === widgetName);
        const languageName = node.widgets.find((w) => w.name === "language_name");
        const languageList = node.widgets.find((w) => w.name === "language_list");

        // 注入隐藏功能
        injectHidden(languageName);
        injectHidden(languageList);

        // 设置初始值
        detectWidget._value = detectWidget.value;

        // 定义属性
        Object.defineProperty(detectWidget, "value", {
            set: function (value) {
                if (value === "choose") {
                    languageName.hidden = true;
                    languageList.hidden = false;
                } else if (value === "input") {
                    languageName.hidden = false;
                    languageList.hidden = true;
                }
                // 重新计算节点大小
                node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]]);
                this._value = value;
            },
            get: function () {
                return this._value;
            }
        });

        // 触发初始状态
        detectWidget.value = detectWidget._value;
    });
}

app.registerExtension({
    name: "ComfyUI-WJNodes.Core",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.name === "load_EasyOCR_model") {
            addCustomLabel_EasyOCR(nodeType, nodeData, "detect");
        }
    }
});