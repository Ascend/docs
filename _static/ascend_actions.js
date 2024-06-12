$(document).ready(function () {
    $.reset_selection = function (elem) {
        elem.parent().children().each(function () {
            $(this).removeClass("selected");
        });
    }

    $.get_options = function () {
        var options = {};
        $('#col-values').children().each(function () {
            var elem = $(this).find(".selected").each(function () {
                var id = $(this).attr("id").split("-");
                var category = id[0];
                var value = id[1];
                if (value === "version") {
                    if (category === "cann")
                        value = $(this).val();
                    else
                        value = $(this).data('version');
                }
                options[category] = value;
            });
        });
        return options;
    }

    $.get_docker_os_versions = function (options) {
        var os_versions = {};
        $.each(docker_images, function (idx, image) {
            var tag = image.split(":")[1];
            var tag_items = tag.split("-");
            var npu_type = tag_items[1];

            var os = tag_items[2];
            var index = os.search(/\d/);
            var os_type = os.substring(0, index);
            var os_version = os.substring(index);

            if (options['os'] === os_type && options['npu'] === npu_type) {
                if (!os_versions[os_type]) {
                    os_versions[os_type] = new Set();
                }
                os_versions[os_type].add(os_version);
            }
        });
        return os_versions;
    }

    $.update_os_verions = function () {
        $("#row-os_version").find("div").not(":first").remove();
        var options = $.get_options();
        // update os_versions
        var os_versions = $.get_docker_os_versions(options);
        var selected_os_versions = os_versions[options['os']];
        if (selected_os_versions == null) {
            $('#row-os_version').append('<div class="values-element-disabled block-1 install-os_version" id="os_version-null" disabled>无可用版本</div>');
        } else {
            var version_length = selected_os_versions.size;
            selected_os_versions.forEach(function (version) {
                $('#row-os_version').append('<div class="values-element block-' + version_length + ' install-os_version" id="os_version-' + version + '">' + version + '</div>');
            });
            $("#row-os_version div:last-child").addClass("selected");
        }
    }

    $.change_options_visible = function () {
        var options = $.get_options();
        if (options['install_type'] === 'direct') {
            $("#header-os_version").hide();
            $("#row-os_version").hide();
        } else {
            $("#header-os_version").show();
            $("#row-os_version").show();
        }
    }

    $.update_cann_versions = function () {
        // reset table.
        var cann_version_select = $('#cann-version');
        cann_version_select.empty();
        cann_version_select.append(new Option("选择CANN的版本", "na"));
        $.reset_selection(cann_version_select);
        $('#driver-version').text("Driver");
        $('#firmware-version').text("Firmware");

        var options = $.get_options();
        // not using docker.
        if (options['install_type'] === "direct") {
            // update select list.
            $.each(package_info, function (key, value) {
                if (options['npu'] in value)
                    cann_version_select.append(new Option("CANN: " + key, key));
            });
        } else {
            $.each(package_info, function (key, value) {
                // not all version has a docker image.
                const tag = key.toLowerCase() + "-" + options['npu'] + "-" + options['os'] + options['os_version'];
                const pkg_info = package_info[key][options['npu']];
                for (const image of docker_images) {
                    if (image.split(":")[1] === tag && pkg_info &&
                        pkg_info.driver_version && pkg_info.firmware_version) {
                        cann_version_select.append(new Option("CANN: " + key, key));
                        break;
                    }
                }
            });
        }
        if (cann_version_select.children().length === 1) {
            cann_version_select.children().first().text('无可用版本');
        }
        cann_version_select.trigger('change');
    }

    $("#col-values").on("click", ".values-element", function () {
        id = $(this).attr("id");
        fields = id.split("-");
        if (id == "cann-version")
            return;

        $.reset_selection($(this));
        $(this).addClass("selected");

        // if os changed, update os version.
        if (fields[0] === "os" || fields[0] === "npu") {
            $.update_os_verions();
        }

        // if install type changed, update options visible.
        if (fields[0] === "install_type") {
            $.change_options_visible();
        }

        // update_cann_version if any option changed.
        $.update_cann_versions();
    });

    $("#col-values").on("change", "select", function () {
        // select cann, driver, formware versions.
        $.reset_selection($(this));
        $('#driver-version').text("Driver");
        $('#firmware-version').text("Firmware");

        if ($(this).val() !== "na") {
            $(this).addClass("selected");
            $('#driver-version').addClass("selected");
            $('#firmware-version').addClass("selected");

            var options = $.get_options();
            var driver_version = package_info[options['cann']][options['npu']].driver_version;
            var firmware_version = package_info[options['cann']][options['npu']].firmware_version;
            $('#driver-version').text("Driver: " + driver_version);
            $('#driver-version').data("version", driver_version);
            $('#firmware-version').text("Firmware: " + firmware_version);
            $('#firmware-version').data("version", firmware_version);
        }
        $.gen_content();
    });

    $.gen_content = function () {
        // instructions need all options selected.
        if ($('#cann-version').val() !== "na") {
            $('#install-instructions').show();
        } else {
            $('#install-instructions').hide();
            return
        }

        var options = $.get_options();

        // install os dependency.
        if (options['os'] === 'ubuntu') {
            $('#install-dependencies-ubuntu').show();
            $('#install-dependencies-openeuler').hide();
        } else {
            $('#install-dependencies-ubuntu').hide();
            $('#install-dependencies-openeuler').show();
        }

        var driver_url = package_info[options['cann']][options['npu']][options['arch']].driver_url;
        var firmware_url = package_info[options['cann']][options['npu']].firmware_url;
        var cann_url = package_info[options['cann']][options['arch']].url;
        var kernel_url = package_info[options['cann']][options['npu']].kernel_url;

        var parts = driver_url.split("/");
        var driver_name = parts[parts.length - 1];
        parts = firmware_url.split("/");
        var firmware_name = parts[parts.length - 1];
        parts = cann_url.split("/");
        var cann_name = parts[parts.length - 1];

        // download and install driver
        $('#install_drvier').html('wget "' + driver_url + '"<br>sudo sh ' + driver_name + ' --full --install-for-all');

        // download and install firmware
        $('#install_firmware').html('wget "' + firmware_url + '"<br>sudo sh ' + firmware_name + ' --full');

        if (options['install_type'] === 'direct') {
            // download and install cann
            $('#install_cann').html('wget "' + cann_url + '"<br>sh ' + cann_name + ' --install');

            // download and install kernel if exist.
            if (kernel_url == null) {
                $('#install_kernel_section').hide();
            }
            else {
                parts = kernel_url.split("/");
                var kernel_name = parts[parts.length - 1];
                $('#install_kernel_section').show();
                // download and install kernel
                $('#install_kernel').html('wget "' + kernel_url + '"<br>sh ' + kernel_name + ' --install');
            }

            $('#use_docker_section').hide();
            $('#install_cann_section').show();
        } else {
            var tag = options['cann'].toLowerCase() + "-" + options['npu'] + "-" + options['os'] + options['os_version'];
            for (var i = 0; i < docker_images.length; i++) {
                if (docker_images[i].split(":")[1] === tag) {
                    var dockerCommand = `
docker run \\
    --name cann_container \\
    --device /dev/davinci1 \\
    --device /dev/davinci_manager \\
    --device /dev/devmm_svm \\
    --device /dev/hisi_hdc \\
    -v /usr/local/dcmi:/usr/local/dcmi \\
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \\
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \\
    -v /etc/ascend_install.info:/etc/ascend_install.info \\
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \\
    -it ${docker_images[i]} bash
                    `;

                    $('#use_docker').html(dockerCommand.trim());
                    break;
                }
            }
            $('#install_cann_section').hide();
            $('#use_docker_section').show();
        }
    }

    $.update_os_verions();
    $.change_options_visible();
    $.update_cann_versions();
});
