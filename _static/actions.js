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
                if (value == "version") {
                    if (category == "cann")
                        value = $(this).val();
                    else
                        value = $(this).data('version');
                }
                options[category] = value;
            });
        });
        return options;
    }

    $.update_os_verions = function () {
        $("#row-os_version").find("div").not(":first").remove();
        var options = $.get_options();
        // update os_versions
        var versions = os_versions[options['os']];
        var version_length = versions.length;
        for (var i = 0; i < version_length; i++) {
            var version = versions[i];
            $('#row-os_version').append('<div class="values-element block-' + version_length + ' install-os_version" id="os_version-' + version + '">' + version + '</div>');
        }
        $("#row-os_version div:last-child").addClass("selected");
    }

    $.update_python_versions = function () {
        $("#row-python_version").find("div").not(":first").remove();
        // update python_versions
        var versions = python_versions;
        var version_length = versions.length;
        for (var i = 0; i < version_length; i++) {
            var version = versions[i];
            $('#row-python_version').append('<div class="values-element block-' + version_length + ' install-python_version" id="python_version-' + version + '">' + version + '</div>');
        }
        $("#row-python_version div:last-child").addClass("selected");
    }

    $.change_options_visible = function () {
        var options = $.get_options();
        if (options['install_type'] == 'direct') {
            $("#header-os_version").hide();
            $("#header-python_version").hide();
            $("#row-os_version").hide();
            $("#row-python_version").hide();
        } else {
            $("#header-os_version").show();
            $("#header-python_version").show();
            $("#row-os_version").show();
            $("#row-python_version").show();
        }
    }

    $has_key = function (obj, path) {
        const keys = path.split('|');
        for (let key of keys) {
            if (!obj || !obj.hasOwnProperty(key)) {
                return false;
            }
            obj = obj[key];
        }
        return true;
    }

    $.update_cann_versions = function () {
        // reset table.
        var cann_version_select = $('#cann-version');
        cann_version_select.empty();
        cann_version_select.append(new Option("Select CANN Version", "na"));
        $.reset_selection(cann_version_select);
        $('#driver-version').text("Driver");
        $('#firmware-version').text("Firmware");

        var options = $.get_options();
        // not using docker.
        if (options['install_type'] == "direct") {
            // update select list.
            $.each(package_info, function (key, value) {
                if (options['npu'] in value)
                    cann_version_select.append(new Option("CANN: " + key, key));
            });
        } else {
            $.each(package_info, function (key, value) {
                // not all version has a docker image.
                var find_key = options['npu'] + "|docker|" + options['os'] + "|" + options['os_version'] + "|" + options['python_version'];
                if ($has_key(value, find_key)) {
                    cann_version_select.append(new Option("CANN: " + key, key));
                }
            });
        }
        cann_version_select.trigger('change');

    }

    $.update_os_verions();
    $.update_python_versions();
    $.change_options_visible();
    $.update_cann_versions();

    $("#col-values").on("click", ".values-element", function () {
        id = $(this).attr("id");
        fields = id.split("-");
        if (fields[1] == "cann_version")
            return;

        $.reset_selection($(this));
        $(this).addClass("selected");

        // if os changed, update os version.
        if (fields[0] == "os") {
            $.update_os_verions();
        }

        // if install type changed, update options visible.
        if (fields[0] == "install_type") {
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

        if ($(this).val() != "na") {
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
        if ($('#cann-version').val() != "na") {
            $('#install-instructions').show();
        } else {
            $('#install-instructions').hide();
            return
        }

        var options = $.get_options();

        // install os dependency.
        if (options['os'] == 'ubuntu') {
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

        if (options['install_type'] == 'direct') {
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
            var docker_image = package_info[options['cann']][options['npu']]['docker'][options['os']][options['os_version']][options['python_version']];
            $('#use_docker').html('docker pull ' + docker_image);
            $('#install_cann_section').hide();
            $('#use_docker_section').show();
        }
    }
});
