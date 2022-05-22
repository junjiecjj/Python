#!/usr/local/bin/lua



local wibox = require("wibox")
local gears = require("gears")
local awful = require("awful")
local cpu_widget = require("awesome-wm-widgets.cpu-widget.cpu-widget")
local logout_menu_widget = require("awesome-wm-widgets.logout-menu-widget.logout-menu")

-- {{{ Wibar
-- Create a textclock widget
mytextclock = wibox.widget.textclock("%a   %H:%M ", 60)

-- Create a wibox for each screen and add it
local taglist_buttons = gears.table.join(
                            awful.button({}, 1, function(t) t:view_only() end),
                            awful.button({modkey}, 1, function(t)
        if client.focus then client.focus:move_to_tag(t) end
    end), awful.button({}, 3, awful.tag.viewtoggle),
                            awful.button({modkey}, 3, function(t)
        if client.focus then client.focus:toggle_tag(t) end
    end), awful.button({}, 4, function(t) awful.tag.viewnext(t.screen) end),
                            awful.button({}, 5, function(t)
        awful.tag.viewprev(t.screen)
    end))

local tasklist_buttons = gears.table.join(
                             awful.button({}, 1, function(c)
        if c == client.focus then
            c.minimized = true
        else
            c:emit_signal("request::activate", "tasklist", {raise = true})
        end
    end), awful.button({}, 3, function()
        awful.menu.client_list({theme = {width = 250}})
    end), awful.button({}, 4, function() awful.client.focus.byidx(1) end),
                             awful.button({}, 5, function()
        awful.client.focus.byidx(-1)
    end))

local function set_wallpaper(s)
    -- Wallpaper
    if beautiful.wallpaper then
        local wallpaper = beautiful.wallpaper
        -- If wallpaper is a function, call it with the screen
        if type(wallpaper) == "function" then wallpaper = wallpaper(s) end
        gears.wallpaper.maximized(wallpaper, s, true)
    end
end

-- Re-set wallpaper when a screen's geometry changes (e.g. different resolution)
-- screen.connect_signal("property::geometry", set_wallpaper)
awful.screen.connect_for_each_screen(function(s)
    -- 下面是tag设置
    --local names = {"@", "+", "", "2", "",""}
    --local names = { "A", "W", "E", "S", "O", "M", "E"}
    local names = { "a", "w", "w", "s", "o", "m", "e"}
    local l = awful.layout.suit -- Just to save some typing: use an alias.
    local layouts = {
        l.tile.left, l.tile.left, l.tile.left, l.floating, l.floating,
        l.floating,l.floating
    }
    awful.tag(names, s, layouts)
    -- Create a promptbox for each screen
    s.mypromptbox = awful.widget.prompt()
    -- Create an imagebox widget which will contain an icon indicating which layout we're using.
    -- We need one layoutbox per screen.
    s.mylayoutbox = awful.widget.layoutbox(s)
    s.mylayoutbox:buttons(gears.table.join(
                              awful.button({}, 1,
                                           function() awful.layout.inc(1) end),
                              awful.button({}, 3,
                                           function()
            awful.layout.inc(-1)
        end), awful.button({}, 4, function() awful.layout.inc(1) end),
                              awful.button({}, 5,
                                           function()
            awful.layout.inc(-1)
        end)))
    -- Create a taglist widget
    s.mytaglist = awful.widget.taglist {
        screen = s,
        filter = awful.widget.taglist.filter.all,
        buttons = taglist_buttons,
        style = {shape = gears.shape.squircle}

    }

    -- Create a tasklist widget
    s.mytasklist = awful.widget.tasklist {
        screen = s,
        filter = awful.widget.tasklist.filter.currenttags,
        buttons = tasklist_buttons,
        style = {
                	border_width = 3,
               	border_color = '#000',
            -- shape = gears.shape.powerline
            -- shape = gears.shape.rectangular_tag
            -- shape = gears.shape.hexagon
            -- shape = gears.shape.rounded_bar
            -- shape = gears.shape.rounded_rect

        },
        layout = {
            spacing = 1,
            spacing_widget = {
                {forced_width = 0, widget = wibox.widget.separator},
                valign = 'right',
                halign = 'center',
                widget = wibox.container.place
            },

            layout = wibox.layout.fixed.horizontal
        },
        widget_template = {
            {
                {
                    {
                        {id = 'icon_role', widget = wibox.widget.imagebox},
                        margins = 0,
                        widget = wibox.container.margin
                    },

                    {id = 'text_role', widget = wibox.widget.textbox},
                    layout = wibox.layout.fixed.horizontal
                },
                left = 20,
                right = 20,
                widget = wibox.container.margin
            },
            id = 'background_role',
            forced_width = 200,
            widget = wibox.container.background
        }

    }
    -- local month_calendar = awful.widget.calendar_popup.month()
    -- month_calendar:attach( mytextclock, 'tr' )
    -- --month_calendar:toggle()
    -- local volumearc_widget = require("awesome-wm-widgets.volumearc-widget.volumearc")
    -- local calendar_widget = require("awesome-wm-widgets.calendar-widget.calendar")
    mytextclock = wibox.widget.textclock("%H:%M   ")
    spacer = wibox.widget.textbox()
    separator = wibox.widget.textbox()
    spacer:set_text(" ")
    separator:set_text("     ")
    -- default

    -- or customized
    -- mytextclock:connect_signal("button::press",
    --    function(_, _, _, button)
    --        if button == 1 then cw.toggle() end
    --    end)
    -- Create the wibox
    s.mywibox = awful.wibar({
        position = "top",
        screen = s,
        height = 28,
        opacity = 0.6,
        --width = 1900,
        --border_width = 5,
        --shape = gears.shape.rounded_rect


    })

    mysystray = wibox.widget.systray()

    -- Add widgets to the wibox
    s.mywibox:setup{
        layout = wibox.layout.align.horizontal,
        align = "centered",
        { -- Left widgets
            layout = wibox.layout.fixed.horizontal,
            s.mytaglist,
            separator
            --s.mypromptbox,
        },
        {
            layout = wibox.layout.fixed.horizontal,
--            s.mytasklist -- Middle widget
        },
        { -- Right widgets
            cpu_widget({
                    width = 70,
                    step_width = 2,
                    step_spacing = 1,
                    --enable_kill_button=true,
                    timeout=5
                    }),
            spacer,
            spacer,
            spacer,
            mysystray,
            spacer,
            spacer,
            mytextclock,
            logout_menu_widget(),
            spacer,
            spacer,
            s.mylayoutbox,
            spacer,
            layout = wibox.layout.fixed.horizontal
        }
    }
end)
-- }}}
