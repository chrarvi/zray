const std = @import("std");
const rl = @import("raylib");
const mu = @import("microui");

pub const FONT_SIZE: c_int = 16;

fn toColor(c: mu.mu_Color) rl.Color {
    return .{ .r = c.r, .g = c.g, .b = c.b, .a = c.a };
}

fn textWidth(font: ?*anyopaque, text: [*c]const u8, text_len: c_int) callconv(.c) c_int {
    _ = font;
    if (text_len < -1) return 0;
    const width = rl.MeasureText(text, FONT_SIZE);
    return width;
}

fn textHeight(font: ?*anyopaque) callconv(.c) c_int {
    _ = font;
    return FONT_SIZE;
}

pub const Gui = struct {
    ctx: mu.mu_Context,

    pub fn init() Gui {
        var out = Gui {
            .ctx = std.mem.zeroes(mu.mu_Context)
        };
        mu.mu_init(&out.ctx);
        out.ctx.text_width = textWidth;
        out.ctx.text_height = textHeight;
        return out;
    }

    pub fn handleInput(self: *Gui) void {
        const pos = rl.GetMousePosition();
        const mx: c_int = @intFromFloat(pos.x);
        const my: c_int = @intFromFloat(pos.y);
        mu.mu_input_mousemove(&self.ctx, mx, my);

        const wheel = rl.GetMouseWheelMove();
        if (wheel != 0) mu.mu_input_scroll(&self.ctx, 0, @as(c_int, @intFromFloat(wheel * -30.0)));

        self.forwardButton(rl.MOUSE_BUTTON_LEFT, mu.MU_MOUSE_LEFT, mx, my);
        self.forwardButton(rl.MOUSE_BUTTON_RIGHT, mu.MU_MOUSE_RIGHT, mx, my);
        self.forwardButton(rl.MOUSE_BUTTON_MIDDLE, mu.MU_MOUSE_MIDDLE, mx, my);
    }

    fn forwardButton(self: *Gui, rl_button: c_int, mu_button: c_int, x: c_int, y: c_int) void {
        if (rl.IsMouseButtonPressed(rl_button)) mu.mu_input_mousedown(&self.ctx, x, y, mu_button);
        if (rl.IsMouseButtonReleased(rl_button)) mu.mu_input_mouseup(&self.ctx, x, y, mu_button);
    }

    pub fn render(self: *Gui) void {
        var cmd: ?*mu.mu_Command = null;
        while (mu.mu_next_command(&self.ctx, &cmd) != 0) {
            const c = cmd orelse break;
            switch (c.type) {
                mu.MU_COMMAND_TEXT => {
                    const c_str: [*c]const u8 = @ptrCast(&c.text.str[0]);
                    rl.DrawText(c_str, c.text.pos.x, c.text.pos.y, FONT_SIZE, toColor(c.text.color));
                },
                mu.MU_COMMAND_RECT => rl.DrawRectangle(c.rect.rect.x, c.rect.rect.y, c.rect.rect.w, c.rect.rect.h, toColor(c.rect.color)),
                mu.MU_COMMAND_ICON => drawIcon(c.icon.id, c.icon.rect, c.icon.color),
                mu.MU_COMMAND_CLIP => {
                    if (c.clip.rect.x == 0 and c.clip.rect.w == 0x1000000 and c.clip.rect.y == 0 and c.clip.rect.h == 0x1000000) {
                        rl.EndScissorMode();
                    } else {
                        rl.BeginScissorMode(c.clip.rect.x, c.clip.rect.y, c.clip.rect.w, c.clip.rect.h);
                    }
                },
                else => {},
            }
        }
    }
};

fn drawIcon(id: c_int, rect: mu.mu_Rect, color: mu.mu_Color) void {
    const c = toColor(color);
    switch (id) {
        mu.MU_ICON_CLOSE => {
            rl.DrawLine(rect.x, rect.y, rect.x + rect.w, rect.y + rect.h, c);
            rl.DrawLine(rect.x + rect.w, rect.y, rect.x, rect.y + rect.h, c);
        },
        mu.MU_ICON_CHECK => {
            rl.DrawLine(rect.x, rect.y + @divTrunc(rect.h, 2), rect.x + @divTrunc(rect.w, 3), rect.y + rect.h, c);
            rl.DrawLine(rect.x + @divTrunc(rect.w, 3), rect.y + rect.h, rect.x + rect.w, rect.y, c);
        },
        mu.MU_ICON_COLLAPSED => rl.DrawTriangle(
            .{ .x = @floatFromInt(rect.x), .y = @floatFromInt(rect.y) },
            .{ .x = @floatFromInt(rect.x), .y = @floatFromInt(rect.y + rect.h) },
            .{ .x = @floatFromInt(rect.x + rect.w), .y = @floatFromInt(rect.y + @divTrunc(rect.h, 2)) },
            c,
        ),
        mu.MU_ICON_EXPANDED => rl.DrawTriangle(
            .{ .x = @floatFromInt(rect.x), .y = @floatFromInt(rect.y) },
            .{ .x = @floatFromInt(rect.x + rect.w), .y = @floatFromInt(rect.y) },
            .{ .x = @floatFromInt(rect.x + @divTrunc(rect.w, 2)), .y = @floatFromInt(rect.y + rect.h) },
            c,
        ),
        else => {},
    }
}
