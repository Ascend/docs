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
                options[category] = value;
            });
        });
        return options;
    }

    $.update_table = function () {
        var options = $.get_options();
    }

    $("#col-values").on("click", ".values-element", function () {
        id = $(this).attr("id");
        fields = id.split("-");

        $.reset_selection($(this));
        $(this).addClass("selected");
        $.update_table();
        $.gen_content();
    });

    
    $.gen_content = function () {
        var options = $.get_options();
        if (options['install_type'] == "docker") {
            $('#install-llamacpp-pip-section').hide();
            $('#install-llamacpp-docker-section').show();
        } else if (options['install_type'] == "pip") {
            $('#install-llamacpp-docker-section').hide();
            $('#install-llamacpp-pip-section').show();
        }
    }

    $.update_table();
    $.gen_content();
});
