$(document).ready(function () {
    var config = null;
    var selectedOptions = {
        version: null,
        product: null,
        cpu_arch: null,
        os: null,
        install_method: null
    };

    function loadConfig() {
        $.getJSON('../../_static/ascend_config.json', function(data) {
            config = data;
            initializeUI();
        }).fail(function() {
            console.error('Failed to load ascend_config.json');
            $('#install-instructions').html('<p style="color: red;">加载配置文件失败，请检查网络连接。</p>');
        });
    }

    function initializeUI() {
        if (!config) return;

        var versions = config.versions;
        var versionKeys = Object.keys(versions);

        var versionHtml = '<div class="mobile-headings">版本</div>';
        versionKeys.forEach(function(key, index) {
            var version = versions[key];
            var selected = index === 0 ? 'selected' : '';
            versionHtml += '<div class="values-element block-' + versionKeys.length + ' install-version ' + selected + '" id="version-' + key + '">' + version.name + '</div>';
        });
        $('#row-version').html(versionHtml);

        if (versionKeys.length > 0) {
            selectedOptions.version = versionKeys[0];
            updateProductSeries();
        }
    }

    function updateProductSeries() {
        if (!config || !selectedOptions.version) return;

        var version = config.versions[selectedOptions.version];
        var productSeries = version.product_series;
        var productKeys = Object.keys(productSeries);

        var productHtml = '<div class="mobile-headings">产品系列</div>';
        productKeys.forEach(function(key, index) {
            var product = productSeries[key];
            var selected = index === 0 ? 'selected' : '';
            productHtml += '<div class="values-element block-' + productKeys.length + ' install-product ' + selected + '" id="product-' + key + '">' + product.name + '</div>';
        });
        $('#row-product').html(productHtml);

        if (productKeys.length > 0) {
            selectedOptions.product = productKeys[0];
            updateCPUArchitectures();
        }
    }

    function updateCPUArchitectures() {
        if (!config || !selectedOptions.version || !selectedOptions.product) return;

        var version = config.versions[selectedOptions.version];
        var productSeries = version.product_series;
        var product = productSeries[selectedOptions.product];
        var cpuArchs = product.cpu_architectures;
        var cpuArchKeys = Object.keys(cpuArchs);

        var cpuArchHtml = '<div class="mobile-headings">CPU架构</div>';
        cpuArchKeys.forEach(function(key, index) {
            var cpuArch = cpuArchs[key];
            var selected = index === 0 ? 'selected' : '';
            cpuArchHtml += '<div class="values-element block-' + cpuArchKeys.length + ' install-cpu_arch ' + selected + '" id="cpu_arch-' + key + '">' + cpuArch.name + '</div>';
        });
        $('#row-cpu_arch').html(cpuArchHtml);

        if (cpuArchKeys.length > 0) {
            selectedOptions.cpu_arch = cpuArchKeys[0];
            updateOperatingSystems();
        }
    }

    function updateOperatingSystems() {
        if (!config || !selectedOptions.version || !selectedOptions.product || !selectedOptions.cpu_arch) return;

        var version = config.versions[selectedOptions.version];
        var productSeries = version.product_series;
        var product = productSeries[selectedOptions.product];
        var cpuArch = product.cpu_architectures[selectedOptions.cpu_arch];
        var operatingSystems = cpuArch.operating_systems;
        var osKeys = Object.keys(operatingSystems);

        var osHtml = '<div class="mobile-headings">操作系统</div>';
        osKeys.forEach(function(key, index) {
            var os = operatingSystems[key];
            var selected = index === 0 ? 'selected' : '';
            osHtml += '<div class="values-element block-' + osKeys.length + ' install-os ' + selected + '" id="os-' + key + '">' + os.name + '</div>';
        });
        $('#row-os').html(osHtml);

        if (osKeys.length > 0) {
            selectedOptions.os = osKeys[0];
            updateInstallMethods();
        }
    }

    function updateInstallMethods() {
        if (!config || !selectedOptions.version || !selectedOptions.product || !selectedOptions.cpu_arch || !selectedOptions.os) return;

        var version = config.versions[selectedOptions.version];
        var productSeries = version.product_series;
        var product = productSeries[selectedOptions.product];
        var cpuArch = product.cpu_architectures[selectedOptions.cpu_arch];
        var operatingSystems = cpuArch.operating_systems[selectedOptions.os];
        var installMethods = operatingSystems.install_methods;
        var installMethodKeys = Object.keys(installMethods);

        var installMethodHtml = '<div class="mobile-headings">安装方式</div>';
        installMethodKeys.forEach(function(key, index) {
            var installMethod = installMethods[key];
            var selected = index === 0 ? 'selected' : '';
            installMethodHtml += '<div class="values-element block-' + installMethodKeys.length + ' install-install_method ' + selected + '" id="install_method-' + key + '">' + installMethod.name + '</div>';
        });
        $('#row-install_method').html(installMethodHtml);

        if (installMethodKeys.length > 0) {
            selectedOptions.install_method = installMethodKeys[0];
            updateInstructions();
        }
    }

    function updateInstructions() {
        if (!config || !selectedOptions.version || !selectedOptions.product || !selectedOptions.cpu_arch || !selectedOptions.os || !selectedOptions.install_method) return;

        var version = config.versions[selectedOptions.version];
        var productSeries = version.product_series;
        var product = productSeries[selectedOptions.product];
        var cpuArch = product.cpu_architectures[selectedOptions.cpu_arch];
        var operatingSystems = cpuArch.operating_systems[selectedOptions.os];
        var installMethod = operatingSystems.install_methods[selectedOptions.install_method];
        var steps = installMethod.steps;

        if (!steps || steps.length === 0) {
            $('#install-instructions').html('<p style="color: red;">暂无安装步骤</p>');
            return;
        }

        var instructionsHtml = '';
        steps.forEach(function(step, index) {
            instructionsHtml += '<section>';
            instructionsHtml += '<h3>' + (index + 1) + '. ' + step.title + '</h3>';
            
            if (step.commands && step.commands.length > 0) {
                step.commands.forEach(function(command) {
                    instructionsHtml += '<div class="highlight-default notranslate">';
                    instructionsHtml += '<div class="highlight">';
                    instructionsHtml += '<pre>' + escapeHtml(command) + '</pre>';
                    instructionsHtml += '</div>';
                    instructionsHtml += '</div>';
                });
            }
            
            instructionsHtml += '</section>';
        });

        $('#install-instructions').html(instructionsHtml);
        $('#install-instructions').show();
    }

    function escapeHtml(text) {
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }

    function resetSelection(elem) {
        elem.parent().children().each(function () {
            $(this).removeClass("selected");
        });
    }

    $("#col-values").on("click", ".values-element", function () {
        var id = $(this).attr("id");
        var fields = id.split("-");

        resetSelection($(this));
        $(this).addClass("selected");

        if (fields[0] === "version") {
            selectedOptions.version = fields[1];
            updateProductSeries();
        } else if (fields[0] === "product") {
            selectedOptions.product = fields[1];
            updateCPUArchitectures();
        } else if (fields[0] === "cpu_arch") {
            selectedOptions.cpu_arch = fields[1];
            updateOperatingSystems();
        } else if (fields[0] === "os") {
            selectedOptions.os = fields[1];
            updateInstallMethods();
        } else if (fields[0] === "install_method") {
            selectedOptions.install_method = fields[1];
            updateInstructions();
        }
    });

    loadConfig();
});
